"""
Locust load testing script for the AI/ML Load Balancer.

This provides a basic smoke test to verify the system works end-to-end.
"""

from locust import HttpUser, task, between
import random


class LoadBalancerUser(HttpUser):
    """Simulates users making requests to the load balancer."""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Initialize user session."""
        self.endpoints = ["/ping", "/cpu", "/io", "/payload"]
        self.weights = [0.6, 0.25, 0.1, 0.05]  # Mixed traffic profile
    
    @task
    def make_request(self):
        """Make a request to a random endpoint."""
        endpoint = random.choices(self.endpoints, weights=self.weights)[0]
        self.client.get(endpoint, name=endpoint)


class SteadyTrafficUser(HttpUser):
    """Generates steady traffic at a constant rate."""
    
    wait_time = between(0.9, 1.1)  # ~1 req/s per user
    
    @task
    def ping(self):
        self.client.get("/ping")
    
    @task(3)
    def cpu(self):
        self.client.get("/cpu")
    
    @task(1)
    def io(self):
        self.client.get("/io")
    
    @task(0.5)
    def payload(self):
        self.client.get("/payload")


class BurstTrafficUser(HttpUser):
    """Generates bursty traffic patterns."""
    
    wait_time = between(0.1, 2.0)  # Variable wait time
    
    def on_start(self):
        self.request_count = 0
    
    @task
    def make_request(self):
        self.request_count += 1
        # Every 20th request, do a burst of 5 rapid requests
        if self.request_count % 20 == 0:
            for _ in range(5):
                self.client.get("/cpu")
                # Very short delay between burst requests
                self.wait_time = between(0.01, 0.05)
        else:
            endpoint = random.choice(["/ping", "/cpu", "/io"])
            self.client.get(endpoint)