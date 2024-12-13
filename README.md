# Auto-Tinder

An AI-powered Tinder automation tool that learns your preferences and automatically swipes based on them.

## Features

- Automatic profile analysis using TensorFlow
- Person detection in images
- Customizable scoring system
- School preference boosting
- Rate-limited API interactions
- Comprehensive test coverage

## Requirements

- Python 3.8+
- Tinder API token
- TensorFlow models (will be downloaded during setup)

## Setup

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# On Windows (CMD):
set TINDER_API_TOKEN=your_token_here

# On Windows (PowerShell):
$env:TINDER_API_TOKEN="your_token_here"

# On Linux/MacOS:
export TINDER_API_TOKEN=your_token_here
```

4. Ensure the required directories exist:
```bash
mkdir -p images/unclassified images/tmp
```

## Usage

1. Run the main script:
```bash
python auto_tinder.py
```

The script will:
- Load the TensorFlow models
- Start analyzing nearby profiles
- Automatically like/dislike based on your preferences
- Run for approximately 2.8 hours by default

## Configuration

Key parameters can be modified in `auto_tinder.py`:
- `LIKE_THRESHOLD`: Minimum score to like a profile (default: 0.8)
- `SCHOOL_BONUS`: Multiplier for preferred schools (default: 1.2)
- `PREFERRED_SCHOOLS`: List of schools that receive bonus points

## Testing

Run the test suite:
```bash
pytest test_auto_tinder.py
```

For test coverage report:
```bash
pytest --cov=. test_auto_tinder.py
```

## Project Structure

```
auto-tinder/
├── auto_tinder.py      # Main script
├── image_classifier.py # Image classification logic
├── person_detector.py  # Person detection in images
├── test_auto_tinder.py # Test suite
├── requirements.txt    # Dependencies
└── images/
    ├── unclassified/  # Temporary storage for unprocessed images
    └── tmp/           # Temporary working directory
```

## Safety Notes

- The script includes rate limiting to prevent API abuse
- Credentials are handled via environment variables for security
- Image processing is done locally
- Temporary files are cleaned up after processing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run the test suite
5. Submit a pull request

## License

MIT License - See LICENSE file for details
