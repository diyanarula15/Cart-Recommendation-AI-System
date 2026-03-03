from locust import HttpUser, task, between

class ZomatoUser(HttpUser):
    wait_time = between(1, 2) 

    @task
    def test_recommendation_api(self):
        payload = {
            "user_id": "16110",
            "cart": ["Bengal Fish Biryani"]
        }
        self.client.post("/api/recommend", json=payload)