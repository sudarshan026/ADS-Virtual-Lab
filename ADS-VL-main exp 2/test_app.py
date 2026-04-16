import unittest
from streamlit.testing.v1 import AppTest

class TestStreamlitApp(unittest.TestCase):
    def test_app_starts(self):
        """Test that the Streamlit app loads and runs the homepage without errors."""
        at = AppTest.from_file("app.py", default_timeout=30)
        at.run()
        
        # Assert there are no exceptions during the initial run
        self.assertFalse(at.exception, f"App threw an exception: {at.exception}")

    def test_dataset_explorer(self):
        """Test switching to Dataset Explorer."""
        at = AppTest.from_file("app.py", default_timeout=30)
        at.run()
        if at.sidebar.radio:
            at.sidebar.radio[0].set_value("📊 Dataset Explorer & EDA").run()
        self.assertFalse(at.exception, f"App threw an exception on Dataset Explorer: {at.exception}")

    def test_ml_classification_lab(self):
        """Test switching to ML Classification Lab."""
        at = AppTest.from_file("app.py", default_timeout=30)
        at.run()
        if at.sidebar.radio:
            at.sidebar.radio[0].set_value("🤖 ML Classification Lab").run()
        self.assertFalse(at.exception, f"App threw an exception on ML Classification Lab: {at.exception}")

    def test_model_results_evaluation(self):
        """Test switching to Model Results & Evaluation."""
        at = AppTest.from_file("app.py", default_timeout=30)
        at.run()
        if at.sidebar.radio:
            at.sidebar.radio[0].set_value("📈 Model Results & Evaluation").run()
        self.assertFalse(at.exception, f"App threw an exception on Model Results: {at.exception}")

    def test_carbon_tracker(self):
        """Test switching to Carbon Footprint Tracker."""
        at = AppTest.from_file("app.py", default_timeout=30)
        at.run()
        if at.sidebar.radio:
            at.sidebar.radio[0].set_value("🌍 Carbon Footprint Tracker").run()
        self.assertFalse(at.exception, f"App threw an exception on Carbon Tracker: {at.exception}")

    def test_quiz(self):
        """Test switching to Quiz page."""
        at = AppTest.from_file("app.py", default_timeout=30)
        at.run()
        if at.sidebar.radio:
            at.sidebar.radio[0].set_value("✅ Quiz").run()
        self.assertFalse(at.exception, f"App threw an exception on Quiz: {at.exception}")

    def test_references(self):
        """Test switching to References page."""
        at = AppTest.from_file("app.py", default_timeout=30)
        at.run()
        if at.sidebar.radio:
            at.sidebar.radio[0].set_value("📚 References").run()
        self.assertFalse(at.exception, f"App threw an exception on References: {at.exception}")

if __name__ == '__main__':
    unittest.main()
