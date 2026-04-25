from flask import Flask, render_template, request, redirect, url_for, session, flash
import os, time, pandas as pd, threading
import concurrent.futures
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import joblib

load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret")

CACHE_DIR = "models"
DASHBOARD_CACHE_FILE = os.path.join(CACHE_DIR, "dashboard_cache.pkl")
BASKET_CACHE_FILE = os.path.join(CACHE_DIR, "basket_cache.pkl")
_CACHE_TTL = 1800  # 30 minutes

_engine = None
def get_engine():
    global _engine
    if _engine is None:
        server   = os.getenv('SQL_SERVER')
        database = os.getenv('SQL_DATABASE')
        user     = os.getenv('SQL_USER')
        password = os.getenv('SQL_PASSWORD')
        
        # Swapped pymssql for pyodbc and specified the Microsoft ODBC Driver
        conn_str = (
            f"mssql+pyodbc://{user}:{password}@{server}/{database}"
            "?driver=ODBC+Driver+18+for+SQL+Server"
        )
        
        _engine = create_engine(
            conn_str,
            pool_pre_ping=True,
            pool_recycle=1800,    
            pool_size=12,         # You can safely bump this back up now!
            max_overflow=6,       
            connect_args={
                "timeout": 120    # Keeps the generous timeout for heavy aggregations
            }
        )
    return _engine

def login_required(f):
    from functools import wraps
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper

# ── Background ML training ─────────────────────────────────────────────────────
# ── Synchronous ML Training ───────────────────────────────────────────────────
def ensure_ml_models():
    """Trains ML models synchronously before the server handles any requests."""
    import os
    if not (os.path.exists("models/clv_model.pkl") and os.path.exists("models/churn_model.pkl")):
        print("Training ML models synchronously on startup...")
        try:
            os.makedirs("models", exist_ok=True)
            from train_models import load_data, build_features, train_clv_model, train_churn_model
            tx, hh = load_data()
            features = build_features(tx, hh)
            train_clv_model(features)
            train_churn_model(features)
            features.to_csv("models/features.csv", index=False)
            print("Startup training complete!")
        except Exception as e:
            print(f"Startup training failed: {e}")

# Run immediately on boot
ensure_ml_models()

# ── Auth ──────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        try:
            with get_engine().connect() as conn:
                row = conn.execute(
                    text("SELECT password_hash FROM users WHERE username=:u"),
                    {"u": username}
                ).fetchone()
            if row and check_password_hash(row[0], password):
                session["user"] = username
                return redirect(url_for("dashboard"))
            flash("Invalid credentials.")
        except Exception as e:
            flash(f"Login error: {e}")
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        try:
            with get_engine().connect() as conn:
                conn.execute(
                    text("INSERT INTO users (username, password_hash, email) VALUES (:u, :p, :e)"),
                    {"u": request.form["username"],
                     "p": generate_password_hash(request.form["password"]),
                     "e": request.form["email"]}
                )
                conn.commit()
            flash("Registered. Please log in.")
            return redirect(url_for("login"))
        except Exception as e:
            flash(f"Registration failed: {e}")
    return render_template("register.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ── Data Pull ─────────────────────────────────────────────────────────────────

@app.route("/data-pull", methods=["GET", "POST"])
@login_required
def data_pull():
    hshd_num = request.form.get("hshd_num", "10")
    show_all = request.form.get("show_all", "false")
    sort_col = request.form.get("sort", "HSHD_NUM")
    sort_dir = request.form.get("sort_dir", "asc")
    page = int(request.form.get("page", 1))
    page_size = 100
    offset = (page - 1) * page_size
    next_dir = "desc" if sort_dir == "asc" else "asc"

    valid_cols = ["HSHD_NUM", "BASKET_NUM", "PURCHASE_DATE", "PRODUCT_NUM",
                  "DEPARTMENT", "COMMODITY", "SPEND", "UNITS"]
    if sort_col not in valid_cols:
        sort_col = "HSHD_NUM"

    order = f"{sort_col} {'DESC' if sort_dir == 'desc' else 'ASC'}"
    
    # FIX 1: Swapped %s to ? for pyodbc parameterization
    where_clause = "" if show_all == "true" else "WHERE t.HSHD_NUM = ?"
    params = None if show_all == "true" else (hshd_num.zfill(4),)

    df = pd.read_sql(f"""
    SELECT
        h.HSHD_NUM, t.BASKET_NUM, t.PURCHASE_DATE, t.PRODUCT_NUM,
        p.DEPARTMENT, p.COMMODITY, t.SPEND, t.UNITS,
        t.STORE_R, t.WEEK_NUM, t.YEAR,
        h.L, h.AGE_RANGE, h.MARITAL, h.INCOME_RANGE,
        h.HOMEOWNER, h.HSHD_COMPOSITION, h.HH_SIZE, h.CHILDREN
    FROM transactions t
    JOIN households h ON t.HSHD_NUM = h.HSHD_NUM
    JOIN products p ON t.PRODUCT_NUM = p.PRODUCT_NUM
    {where_clause}
    ORDER BY {order}
    OFFSET {offset} ROWS FETCH NEXT {page_size} ROWS ONLY
    """, get_engine(), params=params)

    if show_all == "true":
        count_query = "SELECT SUM(row_count) as cnt FROM sys.dm_db_partition_stats WHERE object_id=OBJECT_ID('transactions') AND index_id < 2"
        total = int(pd.read_sql(count_query, get_engine()).iloc[0]['cnt'] or 0)
    else:
        # FIX 2: Swapped %s to ? for pyodbc parameterization
        count_query = "SELECT COUNT(*) as cnt FROM transactions WHERE HSHD_NUM = ?"
        total = pd.read_sql(count_query, get_engine(), params=(hshd_num.zfill(4),)).iloc[0]['cnt']
        
    total_pages = max(1, -(-total // page_size))

    return render_template("data_pull.html",
                           rows=df.to_dict("records"), columns=df.columns.tolist(),
                           hshd_num=hshd_num, show_all=show_all, sort_col=sort_col,
                           sort_dir=sort_dir, next_dir=next_dir, page=page,
                           total_pages=total_pages, total=total)

# ── Upload ────────────────────────────────────────────────────────────────────

@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    if request.method == "POST":
        import tempfile
        import os

        # 1. Grab the uploaded files
        files_to_process = {}
        for key in ["households", "transactions", "products"]:
            f = request.files.get(key)
            if f and f.filename:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
                f.save(tmp.name)
                files_to_process[key] = tmp.name

        # THE FIX: Enforce exactly 3 files
        if len(files_to_process) != 3:
            # Clean up any partial files that were saved before throwing the error
            for p in files_to_process.values():
                if os.path.exists(p):
                    os.unlink(p)
            flash("Upload failed: You must upload all three files (Households, Transactions, and Products) at the same time.")
            return render_template("upload.html")

        try:
            engine = get_engine()
            summary = []
            
            for table, path in files_to_process.items():
                
                # 2. Delete existing data
                with engine.connect() as conn:
                    conn.execute(text(f"DELETE FROM [{table}]"))
                    conn.commit()

                # 3. Load the new CSV into the empty table
                total_rows = 0
                for chunk in pd.read_csv(path, chunksize=50000, low_memory=False):
                    chunk.columns = chunk.columns.str.strip()
                    chunk = chunk.astype(str).replace('nan', None)
                    
                    chunk.to_sql(table, engine, if_exists="append", index=False, chunksize=5000)
                    total_rows += len(chunk)
                    
                summary.append(f"{table}: {total_rows} replaced")

            # 4. Invalidate caches and models
            if os.path.exists(DASHBOARD_CACHE_FILE): os.remove(DASHBOARD_CACHE_FILE)
            if os.path.exists(BASKET_CACHE_FILE): os.remove(BASKET_CACHE_FILE)
            if os.path.exists("models/clv_model.pkl"): os.remove("models/clv_model.pkl")
            if os.path.exists("models/churn_model.pkl"): os.remove("models/churn_model.pkl")
            
            flash("Upload complete — " + " | ".join(summary))
            
        except Exception as e:
            flash(f"Upload failed: {e}")
        finally:
            # Final cleanup
            for p in files_to_process.values():
                if os.path.exists(p):
                    os.unlink(p)
            
    return render_template("upload.html")

# ── Dashboard Logic ───────────────────────────────────────────────────────────

def get_dashboard_data():
    """Fetches dashboard data. Uses a file cache to share data across all 4 Gunicorn workers."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # Check if cache exists and is fresh
    if os.path.exists(DASHBOARD_CACHE_FILE):
        file_age = time.time() - os.path.getmtime(DASHBOARD_CACHE_FILE)
        if file_age < _CACHE_TTL:
            try: return joblib.load(DASHBOARD_CACHE_FILE)
            except: pass

    SQLS = {
        "income": "SELECT h.INCOME_RANGE, AVG(CAST(t.SPEND AS FLOAT)) as avg_spend, COUNT(DISTINCT t.HSHD_NUM) as hh_count FROM transactions t JOIN households h ON t.HSHD_NUM = h.HSHD_NUM WHERE h.INCOME_RANGE NOT IN ('null','') AND h.INCOME_RANGE IS NOT NULL GROUP BY h.INCOME_RANGE ORDER BY avg_spend DESC",
        "hhsize": "SELECT h.HH_SIZE, AVG(CAST(t.SPEND AS FLOAT)) as avg_spend FROM transactions t JOIN households h ON t.HSHD_NUM = h.HSHD_NUM WHERE h.HH_SIZE NOT IN ('null','') AND h.HH_SIZE IS NOT NULL GROUP BY h.HH_SIZE ORDER BY h.HH_SIZE",
        "children": "SELECT CASE WHEN h.CHILDREN = 'null' OR h.CHILDREN IS NULL THEN 'Unknown' WHEN TRY_CAST(h.CHILDREN AS FLOAT) > 0 THEN 'Has Children' ELSE 'No Children' END as children_status, AVG(CAST(t.SPEND AS FLOAT)) as avg_spend, COUNT(DISTINCT t.HSHD_NUM) as hh_count FROM transactions t JOIN households h ON t.HSHD_NUM = h.HSHD_NUM GROUP BY CASE WHEN h.CHILDREN = 'null' OR h.CHILDREN IS NULL THEN 'Unknown' WHEN TRY_CAST(h.CHILDREN AS FLOAT) > 0 THEN 'Has Children' ELSE 'No Children' END",
        "region": "SELECT STORE_R, SUM(CAST(SPEND AS FLOAT)) as total_spend, COUNT(DISTINCT HSHD_NUM) as unique_hh FROM transactions WHERE STORE_R NOT IN ('null','') AND STORE_R IS NOT NULL GROUP BY STORE_R ORDER BY total_spend DESC",
        "weekly": "SELECT CAST(WEEK_NUM AS INT) as WEEK_NUM, CAST(YEAR AS INT) as YEAR, SUM(CAST(SPEND AS FLOAT)) as spend FROM transactions GROUP BY WEEK_NUM, YEAR ORDER BY YEAR, CAST(WEEK_NUM AS INT)",
        "dept_year": "SELECT p.DEPARTMENT, CAST(t.YEAR AS INT) as YEAR, SUM(CAST(t.SPEND AS FLOAT)) as total_spend FROM transactions t JOIN products p ON t.PRODUCT_NUM = p.PRODUCT_NUM WHERE t.YEAR NOT IN ('null','') AND t.YEAR IS NOT NULL GROUP BY p.DEPARTMENT, t.YEAR ORDER BY t.YEAR, total_spend DESC",
        "basket": "WITH top_baskets AS (SELECT TOP 5000 BASKET_NUM FROM transactions GROUP BY BASKET_NUM ORDER BY SUM(CAST(SPEND AS FLOAT)) DESC), basket_comm AS (SELECT DISTINCT t.BASKET_NUM, p.COMMODITY FROM transactions t JOIN products p ON t.PRODUCT_NUM = p.PRODUCT_NUM JOIN top_baskets b ON t.BASKET_NUM = b.BASKET_NUM WHERE p.COMMODITY NOT IN ('null','') AND p.COMMODITY IS NOT NULL) SELECT TOP 10 c1.COMMODITY as item_1, c2.COMMODITY as item_2, COUNT(*) as times_bought_together FROM basket_comm c1 JOIN basket_comm c2 ON c1.BASKET_NUM = c2.BASKET_NUM AND c1.COMMODITY < c2.COMMODITY GROUP BY c1.COMMODITY, c2.COMMODITY ORDER BY times_bought_together DESC",
        "seasonal": "SELECT CASE WHEN CAST(WEEK_NUM AS INT) BETWEEN 1 AND 13 THEN 'Spring' WHEN CAST(WEEK_NUM AS INT) BETWEEN 14 AND 26 THEN 'Summer' WHEN CAST(WEEK_NUM AS INT) BETWEEN 27 AND 39 THEN 'Fall' ELSE 'Winter' END as season, SUM(CAST(SPEND AS FLOAT)) as total_spend, AVG(CAST(SPEND AS FLOAT)) as avg_spend, COUNT(DISTINCT HSHD_NUM) as unique_hh FROM transactions WHERE WEEK_NUM NOT IN ('null','') AND WEEK_NUM IS NOT NULL GROUP BY CASE WHEN CAST(WEEK_NUM AS INT) BETWEEN 1 AND 13 THEN 'Spring' WHEN CAST(WEEK_NUM AS INT) BETWEEN 14 AND 26 THEN 'Summer' WHEN CAST(WEEK_NUM AS INT) BETWEEN 27 AND 39 THEN 'Fall' ELSE 'Winter' END",
        "seasonal_commodity": "SELECT TOP 20 CASE WHEN CAST(t.WEEK_NUM AS INT) BETWEEN 1 AND 13 THEN 'Spring' WHEN CAST(t.WEEK_NUM AS INT) BETWEEN 14 AND 26 THEN 'Summer' WHEN CAST(t.WEEK_NUM AS INT) BETWEEN 27 AND 39 THEN 'Fall' ELSE 'Winter' END as season, p.DEPARTMENT, SUM(CAST(t.SPEND AS FLOAT)) as total_spend FROM transactions t JOIN products p ON t.PRODUCT_NUM = p.PRODUCT_NUM WHERE t.WEEK_NUM NOT IN ('null','') AND t.WEEK_NUM IS NOT NULL GROUP BY CASE WHEN CAST(t.WEEK_NUM AS INT) BETWEEN 1 AND 13 THEN 'Spring' WHEN CAST(t.WEEK_NUM AS INT) BETWEEN 14 AND 26 THEN 'Summer' WHEN CAST(t.WEEK_NUM AS INT) BETWEEN 27 AND 39 THEN 'Fall' ELSE 'Winter' END, p.DEPARTMENT ORDER BY season, total_spend DESC",
        "brand": "SELECT p.BRAND_TY, SUM(CAST(t.SPEND AS FLOAT)) as total_spend, COUNT(*) as transaction_count FROM transactions t JOIN products p ON t.PRODUCT_NUM = p.PRODUCT_NUM WHERE p.BRAND_TY NOT IN ('null','') AND p.BRAND_TY IS NOT NULL GROUP BY p.BRAND_TY",
        "organic": "SELECT p.NATURAL_ORGANIC_FLAG, SUM(CAST(t.SPEND AS FLOAT)) as total_spend, COUNT(DISTINCT t.HSHD_NUM) as buyers FROM transactions t JOIN products p ON t.PRODUCT_NUM = p.PRODUCT_NUM WHERE p.NATURAL_ORGANIC_FLAG NOT IN ('null','') AND p.NATURAL_ORGANIC_FLAG IS NOT NULL GROUP BY p.NATURAL_ORGANIC_FLAG",
        "brand_income": "SELECT h.INCOME_RANGE, p.BRAND_TY, SUM(CAST(t.SPEND AS FLOAT)) as total_spend FROM transactions t JOIN products p ON t.PRODUCT_NUM = p.PRODUCT_NUM JOIN households h ON t.HSHD_NUM = h.HSHD_NUM WHERE p.BRAND_TY NOT IN ('null','') AND p.BRAND_TY IS NOT NULL AND h.INCOME_RANGE NOT IN ('null','') AND h.INCOME_RANGE IS NOT NULL GROUP BY h.INCOME_RANGE, p.BRAND_TY ORDER BY h.INCOME_RANGE, p.BRAND_TY",
        "dept": "SELECT p.DEPARTMENT, SUM(CAST(t.SPEND AS FLOAT)) as total_spend, COUNT(DISTINCT t.HSHD_NUM) as unique_households FROM transactions t JOIN products p ON t.PRODUCT_NUM = p.PRODUCT_NUM GROUP BY p.DEPARTMENT ORDER BY total_spend DESC",
        "commodity": "SELECT TOP 10 p.COMMODITY, SUM(CAST(t.SPEND AS FLOAT)) as total_spend FROM transactions t JOIN products p ON t.PRODUCT_NUM = p.PRODUCT_NUM GROUP BY p.COMMODITY ORDER BY total_spend DESC"
    }

    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        def _q(sql):
            with get_engine().connect() as c:
                return pd.read_sql(sql, c)
        fmap = {ex.submit(_q, sql): name for name, sql in SQLS.items()}
        for fut in concurrent.futures.as_completed(fmap):
            results[fmap[fut]] = fut.result()

    top_clv, churn_counts, avg_clv, high_risk_count = [], {}, 0, 0
    try:
        from ml_models import get_all_predictions
        preds = get_all_predictions()
        top_clv = preds.nlargest(10, 'clv_score')[
            ['HSHD_NUM','clv_score','churn_prob','risk_segment','frequency','recency']
        ].to_dict("records")
        churn_counts    = preds['risk_segment'].value_counts().to_dict()
        avg_clv         = round(preds['clv_score'].mean(), 2)
        high_risk_count = int((preds['risk_segment'] == 'High Risk').sum())
    except Exception as e:
        print(f"ML data not ready for dashboard: {e}")

    cache_data = {
        "income": results.get("income", pd.DataFrame()), "hhsize": results.get("hhsize", pd.DataFrame()), 
        "children": results.get("children", pd.DataFrame()), "region": results.get("region", pd.DataFrame()), 
        "weekly": results.get("weekly", pd.DataFrame()), "dept_year": results.get("dept_year", pd.DataFrame()),
        "basket": results.get("basket", pd.DataFrame()), "seasonal": results.get("seasonal", pd.DataFrame()),
        "seasonal_commodity": results.get("seasonal_commodity", pd.DataFrame()), "brand": results.get("brand", pd.DataFrame()),
        "organic": results.get("organic", pd.DataFrame()), "brand_income": results.get("brand_income", pd.DataFrame()),
        "dept": results.get("dept", pd.DataFrame()), "commodity": results.get("commodity", pd.DataFrame()),
        "top_clv": top_clv, "churn_counts": churn_counts, "avg_clv": avg_clv, "high_risk_count": high_risk_count,
    }

    joblib.dump(cache_data, DASHBOARD_CACHE_FILE)
    return cache_data

@app.route("/api/warm-cache")
def warm_cache():
    """Hidden endpoint for a cron job to keep the dashboard cache fresh."""
    try:
        get_dashboard_data()
        compute_basket_ml()
        return {"status": "success", "message": "Cache warmed."}, 200
    except Exception as e:
        return {"status": "error", "message": str(e)}, 500

@app.route("/dashboard")
@login_required
def dashboard():
    cached = get_dashboard_data()
    return render_template("dashboard.html",
        dept=cached["dept"].to_dict("records"), weekly=cached["weekly"].to_dict("records"),
        income=cached["income"].to_dict("records"), brand=cached["brand"].to_dict("records"),
        organic=cached["organic"].to_dict("records"), commodity=cached["commodity"].to_dict("records"),
        hhsize=cached["hhsize"].to_dict("records"), children=cached["children"].to_dict("records"),
        region=cached["region"].to_dict("records"), dept_year=cached["dept_year"].to_dict("records"),
        basket=cached["basket"].to_dict("records"), seasonal=cached["seasonal"].to_dict("records"),
        seasonal_commodity=cached["seasonal_commodity"].to_dict("records"), brand_income=cached["brand_income"].to_dict("records"),
        top_clv=cached["top_clv"], churn_counts=cached["churn_counts"], avg_clv=cached["avg_clv"], high_risk_count=cached["high_risk_count"]
    )

# ── ML Results & Basket Analysis ──────────────────────────────────────────────

def compute_basket_ml():
    """Train a Gradient Boosting Regressor on commodity-pair co-purchase data."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # Check file cache for other workers
    if os.path.exists(BASKET_CACHE_FILE):
        file_age = time.time() - os.path.getmtime(BASKET_CACHE_FILE)
        if file_age < _CACHE_TTL:
            try: return joblib.load(BASKET_CACHE_FILE)
            except: pass

    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error

    engine = get_engine()
    with engine.connect() as conn:
        pairs = pd.read_sql("""
            WITH top_baskets AS (
                SELECT TOP 5000 BASKET_NUM 
                FROM transactions 
                GROUP BY BASKET_NUM 
                ORDER BY SUM(CAST(SPEND AS DECIMAL(10,2))) DESC
            ),
            basket_comm AS (
                SELECT DISTINCT t.BASKET_NUM, p.COMMODITY
                FROM transactions t
                JOIN products p   ON t.PRODUCT_NUM = p.PRODUCT_NUM
                JOIN top_baskets b ON t.BASKET_NUM  = b.BASKET_NUM
                WHERE p.COMMODITY NOT IN ('null','') AND p.COMMODITY IS NOT NULL
            )
            SELECT TOP 100
                c1.COMMODITY as item_1, c2.COMMODITY as item_2, COUNT(*) as times_bought_together
            FROM basket_comm c1
            JOIN basket_comm c2 ON c1.BASKET_NUM = c2.BASKET_NUM AND c1.COMMODITY < c2.COMMODITY
            GROUP BY c1.COMMODITY, c2.COMMODITY ORDER BY times_bought_together DESC
        """, conn)

        comm_stats = pd.read_sql("""
            SELECT p.COMMODITY, COUNT(DISTINCT t.BASKET_NUM) as basket_count, SUM(CAST(t.SPEND AS FLOAT)) as total_spend
            FROM transactions t JOIN products p ON t.PRODUCT_NUM = p.PRODUCT_NUM
            WHERE p.COMMODITY NOT IN ('null','') AND p.COMMODITY IS NOT NULL
            GROUP BY p.COMMODITY
        """, conn)

    if len(pairs) < 10:
        return None

    freq_map  = comm_stats.set_index('COMMODITY')['basket_count'].to_dict()
    spend_map = comm_stats.set_index('COMMODITY')['total_spend'].to_dict()

    pairs['freq_1']      = pairs['item_1'].map(freq_map).fillna(0)
    pairs['freq_2']      = pairs['item_2'].map(freq_map).fillna(0)
    pairs['spend_1']     = pairs['item_1'].map(spend_map).fillna(0)
    pairs['spend_2']     = pairs['item_2'].map(spend_map).fillna(0)
    pairs['min_freq']    = pairs[['freq_1', 'freq_2']].min(axis=1)
    pairs['max_freq']    = pairs[['freq_1', 'freq_2']].max(axis=1)
    pairs['freq_ratio']  = pairs['min_freq'] / (pairs['max_freq'] + 1)
    pairs['total_spend'] = pairs['spend_1'] + pairs['spend_2']

    feature_cols   = ['freq_1', 'freq_2', 'min_freq', 'max_freq', 'freq_ratio', 'total_spend']
    feature_labels = ['Item 1 Frequency', 'Item 2 Frequency', 'Min Item Frequency', 'Max Item Frequency', 'Frequency Balance', 'Combined Spend']

    X = pairs[feature_cols]
    y = pairs['times_bought_together']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    r2  = round(float(r2_score(y_test, model.predict(X_test))), 3)
    mae = round(float(mean_absolute_error(y_test, model.predict(X_test))), 1)
    pairs['predicted'] = model.predict(X).round(1)

    importances = sorted([{'feature': lbl, 'importance': round(float(v), 4)} for lbl, v in zip(feature_labels, model.feature_importances_)], key=lambda x: -x['importance'])

    result = {
        'pairs': pairs[['item_1', 'item_2', 'times_bought_together', 'predicted']].head(20).to_dict('records'),
        'importances': importances, 'r2': r2, 'mae': mae, 'n_pairs': len(pairs),
    }
    
    joblib.dump(result, BASKET_CACHE_FILE)
    return result

@app.route("/ml")
@login_required
def ml_results():
    all_preds, stats, churn_importances, churn_correlations = [], {}, [], []
    try:
        from ml_models import get_all_predictions, get_churn_importances, get_churn_correlations
        preds = get_all_predictions()
        all_preds = preds.sort_values('churn_prob', ascending=False).to_dict("records")
        stats = {
            "total":       len(preds),
            "high_risk":   int((preds['risk_segment'] == 'High Risk').sum()),
            "medium_risk": int((preds['risk_segment'] == 'Medium Risk').sum()),
            "low_risk":    int((preds['risk_segment'] == 'Low Risk').sum()),
            "avg_clv":     round(preds['clv_score'].mean(), 2),
            "max_clv":     round(preds['clv_score'].max(), 2),
        }
        churn_importances  = get_churn_importances()
        churn_correlations = get_churn_correlations()
    except Exception as e:
        flash(f"ML model error: {e}")

    basket_ml = None
    try:
        basket_ml = compute_basket_ml()
    except Exception as e:
        flash(f"Basket ML error: {e}")

    return render_template("ml.html",
        predictions=all_preds,
        stats=stats,
        churn_importances=churn_importances,
        churn_correlations=churn_correlations,
        basket_ml=basket_ml,
    )
if __name__ == "__main__":
    app.run(debug=True)