from locust import HttpUser, task, between

class ZomatoUser(HttpUser):
    # Simulates a user waiting 1-2 seconds between clicks
    wait_time = between(1, 2) 

    @task
    def test_recommendation_api(self):
        payload = {
            "user_id": "16110",
            "cart": ["Bengal Fish Biryani"]
        }
        # This hits your FastAPI endpoint
        self.client.post("/api/recommend", json=payload)