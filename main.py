from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import time
import os
import pandas as pd
import numpy as np
import traceback

app = FastAPI(title="Zomato")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

def load_data():
    user_files = [f for f in os.listdir('.') if 'users_enriched' in f and f.endswith('.csv')]
    item_files = [f for f in os.listdir('.') if 'items_enriched' in f and f.endswith('.csv')]
    
    u_df, i_df = pd.DataFrame(), pd.DataFrame()
    
    try:
        if user_files and item_files:
            # Load columns as strings to prevent type inference issues
            u_df = pd.read_csv(user_files[0], dtype=str)
            i_df = pd.read_csv(item_files[0])
            
            # Clean and standardize data
            if 'Diet_Type' in u_df.columns:
                u_df['Diet_Type'] = u_df['Diet_Type'].fillna('Vegetarian').astype(str).str.strip()
            
            if 'item_name' in i_df.columns:
                i_df['item_name'] = i_df['item_name'].astype(str).str.strip()
                
            if 'meal_role' in i_df.columns:
                i_df['meal_role'] = i_df['meal_role'].astype(str).str.strip().str.lower()
                role_map = {'bread': 'side', 'sides': 'side', 'starter': 'starter', 'main course': 'main'}
                i_df['meal_role'] = i_df['meal_role'].replace(role_map)
                
            # Ensure critical numeric columns exist and are floats
            for col in ['veg_flag', 'spicy_score', 'order_count', 'approx_price']:
                if col not in i_df.columns:
                    i_df[col] = 0.0 # Default missing columns
                else:
                    i_df[col] = pd.to_numeric(i_df[col], errors='coerce').fillna(0.0)

            print(f"Data Loaded successfully. Users: {len(u_df)}, Items: {len(i_df)}")
        else:
            print("Warning: CSV Files not found in current directory!")
    except Exception as e:
        print(f"Error loading CSVs: {e}")
        
    return u_df, i_df

df_users, df_items = load_data()

@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/recommend")
async def get_recommendations(payload: dict):
    start_time = time.perf_counter()
    
    try:
        # Parse payload
        raw_id = str(payload.get("user_id", "")).split('.')[0].strip()
        raw_cart = payload.get("cart", [])
        
        # Ensure cart is a list
        if not isinstance(raw_cart, list):
            raw_cart = [raw_cart]
            
        cart_clean = [str(i).strip().lower() for i in raw_cart if i]
        
        # User lookup
        user_diet = "Vegetarian" 
        user_spice = "Medium"
        debug_user = "Defaulted to Safe/Veg"

        if not df_users.empty and 'user_id' in df_users.columns and raw_id:
            match = df_users[df_users['user_id'].astype(str) == raw_id]
            if not match.empty:
                row = match.iloc[0]
                user_diet = str(row.get('Diet_Type', 'Vegetarian'))
                user_spice = str(row.get('Spice_Level', 'Medium'))
                debug_user = f"Matched ID: {raw_id} ({user_diet})"

        # Identify cart roles
        cart_roles = []
        if not df_items.empty and 'item_name' in df_items.columns:
            for cart_item_name in cart_clean:
                item_match = df_items[df_items['item_name'].str.lower().str.contains(cart_item_name, regex=False, na=False)]
                if not item_match.empty:
                    role = str(item_match.iloc[0].get('meal_role', 'other')).lower()
                    cart_roles.append(role)

        unique_cart_roles = set(cart_roles)

        # Recommendation scoring
        recommendations = []
        if not df_items.empty:
            scores = df_items.copy()
            scores['final_score'] = 0.0
            scores['reason'] = "AI Affinity Match"

            # Filter by diet
            if "veg" in user_diet.lower():
                scores = scores[scores['veg_flag'] == 1.0].copy()

            # Remove items already in cart
            for item in cart_clean:
                scores = scores[~scores['item_name'].str.lower().str.contains(item, regex=False, na=False)]

            # Exclude current categories (except generic ones)
            for role in unique_cart_roles:
                if role and role != "other":
                    scores = scores[scores['meal_role'] != role].copy()

            # Logic for complementary items
            target = "main"
            if "main" in unique_cart_roles and "side" not in unique_cart_roles:
                target = "side"
                reason = "Added a Side for your Main"
            elif "side" in unique_cart_roles and "main" not in unique_cart_roles:
                target = "main"
                reason = "Added a Main for your Side"
            elif unique_cart_roles:
                target = "other" # dessert/beverage
                reason = "Perfect pairing for your meal"
            else:
                target = "main"
                reason = "Trending in your area"

            # Apply Boost
            scores.loc[scores['meal_role'] == target, 'final_score'] += 10000.0
            scores.loc[scores['meal_role'] == target, 'reason'] = reason

            # Adjust score based on spice preference and popularity
            if user_spice.lower() == "high":
                scores.loc[scores['spicy_score'] > 0.6, 'final_score'] += 100.0
            
            scores['final_score'] += (scores['order_count'] * 0.1)

            # Extract top results
            top_results = scores.nlargest(3, 'final_score')
            for _, row in top_results.iterrows():
                # Cast Numpy types to Python native types
                rec_name = str(row.get('item_name', 'Mystery Item'))
                rec_reason = str(row.get('reason', 'Recommended'))
                # Handle potential NaNs in price
                rec_price = float(row.get('approx_price', 50.0))
                if np.isnan(rec_price):
                    rec_price = 50.0

                recommendations.append({
                    "name": rec_name,
                    "reason": rec_reason,
                    "price": rec_price
                })

        # Fallback recommendations
        if not recommendations:
            recommendations = [{"name": "Chef's Special", "reason": "High Demand Today", "price": 199.0}]

        return {
            "status": "success",
            "debug": {"user": debug_user, "roles_in_cart": list(unique_cart_roles)},
            "recommendations": recommendations,
            "latency_ms": round((time.perf_counter() - start_time) * 1000, 2)
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
        
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "debug": {"error_message": str(e)},
                "recommendations": [
                    {"name": "System Recalibrating", "reason": "Please try again.", "price": 0.0}
                ],
                "latency_ms": 0
            }
        )