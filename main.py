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

# --- 1. PREVENT BROWSER BLOCKING (CORS) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins for local testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

# ==========================================
# 2. FAIL-SAFE DATA LOADER
# ==========================================
def load_data():
    user_files = [f for f in os.listdir('.') if 'users_enriched' in f and f.endswith('.csv')]
    item_files = [f for f in os.listdir('.') if 'items_enriched' in f and f.endswith('.csv')]
    
    u_df, i_df = pd.DataFrame(), pd.DataFrame()
    
    try:
        if user_files and item_files:
            # Explicitly load all columns as strings first to prevent type-crashes
            u_df = pd.read_csv(user_files[0], dtype=str)
            i_df = pd.read_csv(item_files[0])
            
            # Clean and Standardize Data Safely
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
                    i_df[col] = 0.0 # Inject missing columns safely
                else:
                    i_df[col] = pd.to_numeric(i_df[col], errors='coerce').fillna(0.0)

            print(f"✅ DATA LOADED. Users: {len(u_df)}, Items: {len(i_df)}")
        else:
            print("⚠️ WARNING: CSV Files not found in current directory!")
    except Exception as e:
        print(f"❌ FATAL ERROR LOADING CSVs: {e}")
        
    return u_df, i_df

df_users, df_items = load_data()

# ==========================================
# 3. THE EXCEPTION-PROOF ENDPOINT
# ==========================================

@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/recommend")
async def get_recommendations(payload: dict):
    start_time = time.perf_counter()
    
    # GLOBAL TRY-CATCH: Guarantees the frontend never hangs
    try:
        # --- A. SAFE PAYLOAD PARSING ---
        raw_id = str(payload.get("user_id", "")).split('.')[0].strip()
        raw_cart = payload.get("cart", [])
        
        # Guard against someone passing a string instead of a list for the cart
        if not isinstance(raw_cart, list):
            raw_cart = [raw_cart]
            
        cart_clean = [str(i).strip().lower() for i in raw_cart if i]
        
        # --- B. USER LOOKUP WITH BOUNDS CHECKING ---
        user_diet = "Vegetarian" 
        user_spice = "Medium"
        debug_user = "Defaulted to Safe/Veg"

        if not df_users.empty and 'user_id' in df_users.columns and raw_id:
            # Explicit string comparison
            match = df_users[df_users['user_id'].astype(str) == raw_id]
            if not match.empty:
                row = match.iloc[0]
                user_diet = str(row.get('Diet_Type', 'Vegetarian'))
                user_spice = str(row.get('Spice_Level', 'Medium'))
                debug_user = f"Matched ID: {raw_id} ({user_diet})"

        # --- C. CART ROLE IDENTIFICATION ---
        cart_roles = []
        if not df_items.empty and 'item_name' in df_items.columns:
            for cart_item_name in cart_clean:
                # Fuzzy matching
                item_match = df_items[df_items['item_name'].str.lower().str.contains(cart_item_name, regex=False, na=False)]
                if not item_match.empty:
                    role = str(item_match.iloc[0].get('meal_role', 'other')).lower()
                    cart_roles.append(role)

        unique_cart_roles = set(cart_roles)

        # --- D. SCORING ENGINE ---
        recommendations = []
        if not df_items.empty:
            scores = df_items.copy()
            scores['final_score'] = 0.0
            scores['reason'] = "AI Affinity Match"

            # 1. DIET NUKE
            if "veg" in user_diet.lower():
                scores = scores[scores['veg_flag'] == 1.0].copy()

            # 2. PREVENT EXACT DUPLICATES
            for item in cart_clean:
                scores = scores[~scores['item_name'].str.lower().str.contains(item, regex=False, na=False)]

            # 3. CATEGORY EXCLUSION (Bread for Bread fix)
            for role in unique_cart_roles:
                if role and role != "other": # Don't exclude beverages/desserts
                    scores = scores[scores['meal_role'] != role].copy()

            # 4. MEAL SEQUENCE LOGIC
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

            # 5. SPICE & POPULARITY
            if user_spice.lower() == "high":
                scores.loc[scores['spicy_score'] > 0.6, 'final_score'] += 100.0
            
            scores['final_score'] += (scores['order_count'] * 0.1)

            # 6. JSON-SAFE EXTRACTION (CRITICAL FIX)
            top_results = scores.nlargest(3, 'final_score')
            for _, row in top_results.iterrows():
                # Explicitly cast Numpy types to Python native types so JSON doesn't crash
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

        # Fallback if catalog is empty or fully filtered
        if not recommendations:
            recommendations = [{"name": "Chef's Special", "reason": "High Demand Today", "price": 199.0}]

        return {
            "status": "success",
            "debug": {"user": debug_user, "roles_in_cart": list(unique_cart_roles)},
            "recommendations": recommendations,
            "latency_ms": round((time.perf_counter() - start_time) * 1000, 2)
        }

    except Exception as e:
        # IF ANYTHING CRASHES, DO NOT HANG. RETURN AN ERROR JSON!
        print("\n" + "="*50)
        print("🚨 API CRASH DETECTED 🚨")
        traceback.print_exc() # This prints the exact line number of the error to your terminal
        print("="*50 + "\n")
        
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