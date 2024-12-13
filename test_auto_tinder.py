import pytest
from unittest.mock import patch, MagicMock
import datetime
from auto_tinder import TinderAPI, Person, Profile

class TestTinderAPI:
    @pytest.fixture
    def api(self):
        return TinderAPI("test-token")

    def test_init(self, api):
        assert api._token == "test-token"
        assert api._headers == {"X-Auth-Token": "test-token"}
        assert api._rate_limit_delay == 1.0

    @patch('requests.get')
    def test_profile_retrieval(self, mock_get):
        # Mock the response from the Tinder API
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": {
                "user": {
                    "_id": "12345",
                    "name": "Test User",
                    "bio": "This is a test bio",
                    "birth_date": "1990-01-01T00:00:00.000Z",
                    "gender": 0,
                    "photos": [{"url": "http://example.com/photo1.jpg"}],
                    "jobs": [],
                    "schools": []
                },
                "account": {
                    "email": "test@example.com",
                    "account_phone_number": "1234567890"
                }
            }
        }
        mock_get.return_value = mock_response
        mock_response.raise_for_status.return_value = None

        api = TinderAPI("test-token")
        profile = api.profile()

        assert isinstance(profile, Profile)
        assert profile.id == "12345"
        assert profile.name == "Test User"
        assert profile.email == "test@example.com"

    @patch('requests.get')
    def test_nearby_persons(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": {
                "results": [
                    {
                        "user": {
                            "_id": "user1",
                            "name": "Person 1",
                            "photos": []
                        }
                    },
                    {
                        "user": {
                            "_id": "user2",
                            "name": "Person 2",
                            "photos": []
                        }
                    }
                ]
            }
        }
        mock_get.return_value = mock_response
        mock_response.raise_for_status.return_value = None

        api = TinderAPI("test-token")
        persons = api.nearby_persons()

        assert len(persons) == 2
        assert all(isinstance(p, Person) for p in persons)
        assert persons[0].id == "user1"
        assert persons[1].id == "user2"

class TestPerson:
    @pytest.fixture
    def api(self):
        return MagicMock()

    def test_person_initialization_minimal_data(self, api):
        data = {
            "_id": "12345",
            "name": "Test User"
        }
        person = Person(data, api)

        assert person.id == "12345"
        assert person.name == "Test User"
        assert person.bio == ""
        assert person.birth_date is None
        assert person.images == []
        assert person.jobs == []
        assert person.schools == []

    def test_person_initialization_full_data(self, api):
        data = {
            "_id": "12345",
            "name": "Test User",
            "bio": "Test bio",
            "birth_date": "1990-01-01T00:00:00.000Z",
            "gender": 0,
            "photos": [{"url": "http://example.com/photo1.jpg"}],
            "jobs": [{"title": {"name": "Developer"}, "company": {"name": "Tech Corp"}}],
            "schools": [{"name": "Test University"}]
        }
        person = Person(data, api)

        assert person.id == "12345"
        assert person.name == "Test User"
        assert person.bio == "Test bio"
        assert isinstance(person.birth_date, datetime.datetime)
        assert person.gender == "Male"
        assert person.images == ["http://example.com/photo1.jpg"]
        assert person.jobs == [{"title": "Developer", "company": "Tech Corp"}]
        assert person.schools == ["Test University"]

    @patch('requests.get')
    def test_download_images(self, mock_get, api, tmp_path):
        mock_response = MagicMock()
        mock_response.content = b"fake_image_data"
        mock_get.return_value = mock_response
        mock_response.raise_for_status.return_value = None

        data = {
            "_id": "12345",
            "name": "Test User",
            "photos": [{"url": "http://example.com/photo1.jpg"}]
        }
        person = Person(data, api)
        person.download_images(folder=str(tmp_path))

        # Check if image was downloaded
        downloaded_files = list(tmp_path.glob("*.jpeg"))
        assert len(downloaded_files) == 1
        assert downloaded_files[0].name.startswith("12345_Test User_0")

class TestProfile(TestPerson):
    def test_profile_initialization(self, api):
        data = {
            "user": {
                "_id": "12345",
                "name": "Test User",
                "age_filter_min": 20,
                "age_filter_max": 30,
                "distance_filter": 10,
                "gender_filter": 1
            },
            "account": {
                "email": "test@example.com",
                "account_phone_number": "1234567890"
            }
        }
        profile = Profile(data, api)

        assert profile.id == "12345"
        assert profile.name == "Test User"
        assert profile.email == "test@example.com"
        assert profile.phone_number == "1234567890"
        assert profile.age_min == 20
        assert profile.age_max == 30
        assert profile.max_distance == 10
        assert profile.gender_filter == "Female"

if __name__ == "__main__":
    pytest.main([__file__])
