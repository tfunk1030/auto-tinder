import requests
import datetime
from geopy.geocoders import Nominatim
from time import sleep, time
from random import random
import tensorflow as tf
from likeliness_classifier import Classifier
import person_detector
import os
from typing import List, Dict, Optional

TINDER_URL = "https://api.gotinder.com"
geolocator = Nominatim(user_agent="auto-tinder")
PROF_FILE = "./images/unclassified/profiles.txt"
TEMP_IMAGE_PATH = "./images/tmp/run.jpg"

class TinderAPI:
    def __init__(self, token: str):
        self._token = token
        self._headers = {"X-Auth-Token": self._token}
        self._rate_limit_delay = 1.0  # Delay between API calls in seconds

    def _make_request(self, endpoint: str, method: str = "GET") -> dict:
        """Make a rate-limited request to the Tinder API"""
        sleep(self._rate_limit_delay)
        url = f"{TINDER_URL}{endpoint}"
        response = requests.get(url, headers=self._headers)
        response.raise_for_status()
        return response.json()

    def profile(self) -> 'Profile':
        """Get the user's profile"""
        data = self._make_request("/v2/profile?include=account%2Cuser")
        return Profile(data["data"], self)

    def matches(self, limit: int = 10) -> List['Person']:
        """Get recent matches"""
        data = self._make_request(f"/v2/matches?count={limit}")
        return [Person(match["person"], self) for match in data["data"]["matches"]]

    def like(self, user_id: str) -> Dict[str, bool]:
        """Like a user"""
        data = self._make_request(f"/like/{user_id}")
        return {
            "is_match": data["match"],
            "liked_remaining": data["likes_remaining"]
        }

    def dislike(self, user_id: str) -> bool:
        """Dislike a user"""
        self._make_request(f"/pass/{user_id}")
        return True

    def nearby_persons(self) -> List['Person']:
        """Get nearby persons"""
        data = self._make_request("/v2/recs/core")
        return [Person(user["user"], self) for user in data["data"]["results"]]

class Person:
    def __init__(self, data: dict, api: TinderAPI):
        self._api = api
        self.id = data["_id"]
        self.name = data.get("name", "Unknown")
        self.bio = data.get("bio", "")
        self.distance = data.get("distance_mi", 0) / 1.60934

        self.birth_date = None
        if birth_date_str := data.get("birth_date"):
            try:
                self.birth_date = datetime.datetime.strptime(birth_date_str, '%Y-%m-%dT%H:%M:%S.%fZ')
            except ValueError:
                pass

        self.gender = ["Male", "Female", "Unknown"][data.get("gender", 2)]
        self.images = [photo["url"] for photo in data.get("photos", [])]
        
        self.jobs = [
            {
                "title": job.get("title", {}).get("name"),
                "company": job.get("company", {}).get("name")
            }
            for job in data.get("jobs", [])
        ]
        
        self.schools = [school["name"] for school in data.get("schools", [])]

        if pos := data.get("pos"):
            try:
                self.location = geolocator.reverse(f'{pos["lat"]}, {pos["lon"]}')
            except Exception:
                self.location = None

    def __repr__(self) -> str:
        birth_date_str = self.birth_date.strftime('%d.%m.%Y') if self.birth_date else "Unknown"
        return f"{self.id} - {self.name} ({birth_date_str})"

    def like(self) -> Dict[str, bool]:
        return self._api.like(self.id)

    def dislike(self) -> bool:
        return self._api.dislike(self.id)

    def download_images(self, folder: str = ".", sleep_max_for: float = 0) -> None:
        """Download person's images with rate limiting"""
        os.makedirs(folder, exist_ok=True)
        
        # Check if profile was already processed
        if os.path.exists(PROF_FILE):
            with open(PROF_FILE, "r") as f:
                if self.id in f.read().splitlines():
                    return

        # Mark profile as processed
        with open(PROF_FILE, "a") as f:
            f.write(f"{self.id}\n")

        # Download images
        for index, image_url in enumerate(self.images):
            try:
                response = requests.get(image_url, stream=True)
                response.raise_for_status()
                
                image_path = os.path.join(folder, f"{self.id}_{self.name}_{index}.jpeg")
                with open(image_path, "wb") as f:
                    f.write(response.content)
                
                if sleep_max_for > 0:
                    sleep(random() * sleep_max_for)
                    
            except requests.RequestException as e:
                print(f"Failed to download image {image_url}: {e}")

    def predict_likeliness(self, classifier: Classifier, sess) -> float:
        """Predict likeliness score for the person"""
        ratings = []
        
        for image_url in self.images:
            try:
                # Download image
                response = requests.get(image_url, stream=True)
                response.raise_for_status()
                
                os.makedirs(os.path.dirname(TEMP_IMAGE_PATH), exist_ok=True)
                with open(TEMP_IMAGE_PATH, "wb") as f:
                    f.write(response.content)

                # Detect and process person in image
                if img := person_detector.get_person(TEMP_IMAGE_PATH, sess):
                    img = img.convert('L')
                    img.save(TEMP_IMAGE_PATH, "jpeg")
                    
                    certainty = classifier.classify(TEMP_IMAGE_PATH)
                    ratings.append(certainty["positive"])
                    
            except Exception as e:
                print(f"Error processing image {image_url}: {e}")
                continue

        # Calculate final score
        if not ratings:
            return 0.001
            
        ratings.sort(reverse=True)
        ratings = ratings[:5]  # Take top 5 ratings
        
        if len(ratings) == 1:
            return ratings[0]
            
        return ratings[0] * 0.6 + sum(ratings[1:]) / len(ratings[1:]) * 0.4

class Profile(Person):
    def __init__(self, data: dict, api: TinderAPI):
        super().__init__(data["user"], api)
        
        account = data["account"]
        self.email = account.get("email")
        self.phone_number = account.get("account_phone_number")
        
        user = data["user"]
        self.age_min = user["age_filter_min"]
        self.age_max = user["age_filter_max"]
        self.max_distance = user["distance_filter"]
        self.gender_filter = ["Male", "Female"][user["gender_filter"]]

def main():
    # Get API token from environment variable
    token = os.getenv("TINDER_API_TOKEN")
    if not token:
        raise ValueError("TINDER_API_TOKEN environment variable not set")

    # Initialize API
    api = TinderAPI(token)

    # Load models
    detection_graph = person_detector.open_graph()
    with detection_graph.as_default():
        with tf.Session() as sess:
            classifier = Classifier(
                graph="./tf/training_output/retrained_graph.pb",
                labels="./tf/training_output/retrained_labels.txt"
            )

            try:
                run_auto_tinder(api, classifier, sess)
            finally:
                classifier.close()

def run_auto_tinder(api: TinderAPI, classifier: Classifier, sess) -> None:
    """Main auto-tinder loop"""
    # Load school preferences from configuration
    PREFERRED_SCHOOLS = [
        "Universität Zürich", "University of Zurich", "UZH",
        "HWZ Hochschule für Wirtschaft Zürich", "ETH Zürich",
        "ETH Zurich", "ETH", "ETHZ", "Hochschule Luzern",
        "HSLU", "ZHAW", "Zürcher Hochschule für Angewandte Wissenschaften",
        "Universität Bern", "Uni Bern", "PHLU", "PH Luzern",
        "Fachhochschule Luzern", "Eidgenössische Technische Hochschule Zürich"
    ]

    LIKE_THRESHOLD = 0.8
    SCHOOL_BONUS = 1.2
    
    end_time = time() + 60 * 60 * 2.8  # Run for 2.8 hours
    
    while time() < end_time:
        try:
            print(f"------ TIME LEFT: {(end_time - time())/60:.1f} min -----")
            
            for person in api.nearby_persons():
                try:
                    # Calculate base score
                    score = person.predict_likeliness(classifier, sess)
                    
                    # Apply school bonus
                    if any(school in person.schools for school in PREFERRED_SCHOOLS):
                        score *= SCHOOL_BONUS
                    
                    # Log person details
                    print("\n-------------------------")
                    print(f"ID: {person.id}")
                    print(f"Name: {person.name}")
                    print(f"Schools: {person.schools}")
                    print(f"Score: {score:.3f}")
                    
                    # Make decision
                    if score > LIKE_THRESHOLD:
                        result = person.like()
                        print("LIKE")
                        print(f"Response: {result}")
                    else:
                        person.dislike()
                        print("DISLIKE")
                        
                except Exception as e:
                    print(f"Error processing person {person.id}: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error in main loop: {e}")
            sleep(5)  # Wait before retrying

if __name__ == "__main__":
    main()
