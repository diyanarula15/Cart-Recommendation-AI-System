# Zomato Recommendation Engine: Hyper-Personalised Add-on Ranker

## 🚀 Overview
A production-ready recommendation system built to suggest relevant add-on items (sides, drinks, desserts) based on live cart composition, environmental context, and historical user behavior.

### Key Features
* **Dual-Brain Architecture**: Combines a Transformer for sequential meal "grammar" and a Deep & Cross Network (DCN) for complex contextual feature interactions.
* **Zero-History Cold Start**: Uses an LLM-augmented taxonomy to recommend brand-new items or serve users with no prior order history.
* **Production-Optimized**: Achieves ~16.75ms total latency using Redis for feature storage and ONNX for model inference.
* **Diversity Guardrails**: Implements taxonomy-aware penalization to ensure varied recommendations (e.g., avoiding a list of five different breads).

## 🛠️ Tech Stack
* **Model**: Transformer + Deep & Cross Network (DCN).
* **Backend**: FastAPI for high-performance API serving.
* **Feature Store**: Redis for ultra-fast, in-memory user and item embedding retrieval.
* **Optimization**: ONNX runtime for accelerated CPU inference.
* **Infrastructure**: Containerized with Docker and scalable via Kubernetes.

## 📊 Performance Metrics
* **Latency**: Sub-300ms SLA (Actual: ~16.75ms).
* **Recall@5**: 0.9833 (True add-ons are almost always in the top 5).
* **NDCG@5**: 0.8014 (Ensures top items are ranked correctly).
* **Throughput**: Successfully stress-tested at 300+ Requests Per Second (RPS).

## 🧠 Core Logic
1. **Transformer Stream**: Acts as "Chef's intuition," identifying missing meal roles (e.g., if you have Butter Chicken, it prioritizes Garlic Naan over another main).
2. **DCN Stream**: Calculates interactions like `User(Spice Lover) + Time(Friday Night) + Cart(High Value) = Premium Spicy Appetiser`.
3. **Gap Analysis**: Dynamically updates a "Context Vector" as items are added to identify missing categories like Beverages or Desserts.

## 📁 Data Strategy
The model was trained on a "Digital Twin" of the Indian food ecosystem, derived from three core Kaggle datasets:
* **Indian Takeaway Orders**: Transactional co-occurrence patterns.
* **Indian Food 101**: Culinary taxonomy and flavor profiles.
* **Restaurant Order Details**: Temporal and user-level behavior.
