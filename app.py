from flask import Flask, render_template, request, redirect, url_for, session, flash
import os, time, pandas as pd
import concurrent.futures
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

_dashboard_cache  = {}
_basket_ml_cache  = {}
_CACHE_TTL = 1800  # 30 minutes

def compute_basket_ml():
    """Train a Gradient Boosting Regressor on commodity-pair co-purchase data."""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error

    cached = _basket_ml_cache.get("data")
    if cached and (time.time() - cached["ts"]) < 600:
        return cached["result"]

    engine = get_engine()
    with engine.connect() as conn:
        pairs = pd.read_sql("""
            SELECT TOP 100
                c1.COMMODITY as item_1,
                c2.COMMODITY as item_2,
                COUNT(*) as times_bought_together
            FROM (SELECT DISTINCT t.BASKET_NUM, p.COMMODITY
                  FROM transactions t JOIN products p ON t.PRODUCT_NUM = p.PRODUCT_NUM
                  WHERE p.COMMODITY NOT IN ('null','') AND p.COMMODITY IS NOT NULL) c1
            JOIN (SELECT DISTINCT t.BASKET_NUM, p.COMMODITY
                  FROM transactions t JOIN products p ON t.PRODUCT_NUM = p.PRODUCT_NUM
                  WHERE p.COMMODITY NOT IN ('null','') AND p.COMMODITY IS NOT NULL) c2
                ON c1.BASKET_NUM = c2.BASKET_NUM AND c1.COMMODITY < c2.COMMODITY
            GROUP BY c1.COMMODITY, c2.COMMODITY
            ORDER BY times_bought_together DESC
        """, conn)

        comm_stats = pd.read_sql("""
            SELECT p.COMMODITY,
                   COUNT(DISTINCT t.BASKET_NUM) as basket_count,
                   SUM(CAST(t.SPEND AS FLOAT))  as total_spend
            FROM transactions t
            JOIN products p ON t.PRODUCT_NUM = p.PRODUCT_NUM
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
    feature_labels = ['Item 1 Frequency', 'Item 2 Frequency',
                      'Min Item Frequency', 'Max Item Frequency',
                      'Frequency Balance', 'Combined Spend']

    X = pairs[feature_cols]
    y = pairs['times_bought_together']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                      max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2  = round(float(r2_score(y_test, y_pred)), 3)
    mae = round(float(mean_absolute_error(y_test, y_pred)), 1)

    pairs['predicted'] = model.predict(X).round(1)

    importances = sorted(
        [{'feature': lbl, 'importance': round(float(v), 4)}
         for lbl, v in zip(feature_labels, model.feature_importances_)],
        key=lambda x: -x['importance']
    )

    result = {
        'pairs':      pairs[['item_1', 'item_2', 'times_bought_together', 'predicted']]
                           .head(20).to_dict('records'),
        'importances': importances,
        'r2':          r2,
        'mae':         mae,
        'n_pairs':     len(pairs),
    }
    _basket_ml_cache["data"] = {"ts": time.time(), "result": result}
    return result

load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret")

_engine = None
def get_engine():
    global _engine
    if _engine is None:
        server   = os.getenv('SQL_SERVER')
        database = os.getenv('SQL_DATABASE')
        user     = os.getenv('SQL_USER')
        password = os.getenv('SQL_PASSWORD')
        _engine = create_engine(
            f"mssql+pymssql://{user}:{password}@{server}/{database}",
            pool_pre_ping=True,   # detect & drop stale connections before use
            pool_recycle=1800,    # recycle every 30 min (Azure kills idle after ~30 min)
            pool_size=3,          # keep 3 persistent connections warm
            max_overflow=5,       # allow 5 extra under burst load
            connect_args={"timeout": 30, "login_timeout": 30}
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

# ── Data Pull (Req 3 & 4) ─────────────────────────────────────────────────────

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
    where_clause = "" if show_all == "true" else "WHERE t.HSHD_NUM = %s"
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

    count_query = "SELECT COUNT(*) as cnt FROM transactions" if show_all == "true" \
                  else "SELECT COUNT(*) as cnt FROM transactions WHERE HSHD_NUM = %s"
    count_params = None if show_all == "true" else (hshd_num.zfill(4),)
    total = pd.read_sql(count_query, get_engine(), params=count_params).iloc[0]['cnt']
    total_pages = max(1, -(-total // page_size))

    return render_template("data_pull.html",
                           rows=df.to_dict("records"),
                           columns=df.columns.tolist(),
                           hshd_num=hshd_num,
                           show_all=show_all,
                           sort_col=sort_col,
                           sort_dir=sort_dir,
                           next_dir=next_dir,
                           page=page,
                           total_pages=total_pages,
                           total=total)

# ── Upload (Req 5) ────────────────────────────────────────────────────────────

@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    if request.method == "POST":
        import tempfile

        # Primary key(s) used to detect duplicates for each table
        PKS = {
            "households":   ["HSHD_NUM"],
            "products":     ["PRODUCT_NUM"],
            "transactions": ["BASKET_NUM", "PRODUCT_NUM"],
        }

        files = {}
        for key in ["households", "transactions", "products"]:
            f = request.files.get(key)
            if f:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
                f.save(tmp.name)
                files[key] = tmp.name

        if len(files) == 3:
            try:
                engine = get_engine()
                summary = []
                for table, path in [("households",   files["households"]),
                                     ("products",     files["products"]),
                                     ("transactions", files["transactions"])]:
                    df = pd.read_csv(path)
                    df.columns = df.columns.str.strip()
                    df = df.astype(str).replace('nan', None)

                    pk_cols = PKS[table]
                    staging = f"_stage_{table}"
                    cols     = ", ".join(f"s.[{c}]" for c in df.columns)
                    pk_join  = " AND ".join(f"m.[{c}] = s.[{c}]" for c in pk_cols)

                    # Bulk-load CSV into a staging table (server-side dedup next)
                    df.to_sql(staging, engine, if_exists="replace",
                              index=False, chunksize=500)

                    with engine.connect() as conn:
                        result = conn.execute(text(f"""
                            INSERT INTO [{table}] ({', '.join(f'[{c}]' for c in df.columns)})
                            SELECT {cols}
                            FROM   [{staging}] s
                            WHERE  NOT EXISTS (
                                SELECT 1 FROM [{table}] m WHERE {pk_join}
                            )
                        """))
                        inserted = result.rowcount
                        conn.execute(text(f"DROP TABLE [{staging}]"))
                        conn.commit()

                    summary.append(
                        f"{table}: {inserted} new, "
                        f"{len(df) - inserted} skipped (already exist)"
                    )

                _dashboard_cache.clear()
                flash("Upload complete — " + " | ".join(summary))
            except Exception as e:
                flash(f"Upload failed: {e}")
            finally:
                for p in files.values():
                    os.unlink(p)
        else:
            flash("Please upload all three files.")
    return render_template("upload.html")

# ── Dashboard (Req 6) ─────────────────────────────────────────────────────────

@app.route("/dashboard")
@login_required
def dashboard():
    engine = get_engine()

    cached = _dashboard_cache.get("data")
    if cached and (time.time() - cached["ts"]) < _CACHE_TTL:
        income          = cached["income"]
        hhsize          = cached["hhsize"]
        children        = cached["children"]
        region          = cached["region"]
        weekly          = cached["weekly"]
        dept_year       = cached["dept_year"]
        basket          = cached["basket"]
        seasonal        = cached["seasonal"]
        seasonal_commodity = cached["seasonal_commodity"]
        brand           = cached["brand"]
        organic         = cached["organic"]
        brand_income    = cached["brand_income"]
        dept            = cached["dept"]
        commodity       = cached["commodity"]
        top_clv         = cached.get("top_clv", [])
        churn_counts    = cached.get("churn_counts", {})
        avg_clv         = cached.get("avg_clv", 0)
        high_risk_count = cached.get("high_risk_count", 0)
    else:
        # ── Run all 14 queries in parallel ────────────────────────────────────
        def _q(sql):
            """Each call gets its own pooled connection so queries run concurrently."""
            with get_engine().connect() as c:
                return pd.read_sql(sql, c)

        SQLS = {
            "income": """
                SELECT h.INCOME_RANGE,
                       AVG(CAST(t.SPEND AS DECIMAL(10,2))) as avg_spend,
                       COUNT(DISTINCT t.HSHD_NUM) as hh_count
                FROM transactions t JOIN households h ON t.HSHD_NUM = h.HSHD_NUM
                WHERE h.INCOME_RANGE NOT IN ('null','') AND h.INCOME_RANGE IS NOT NULL
                GROUP BY h.INCOME_RANGE ORDER BY avg_spend DESC
            """,
            "hhsize": """
                SELECT h.HH_SIZE, AVG(CAST(t.SPEND AS DECIMAL(10,2))) as avg_spend
                FROM transactions t JOIN households h ON t.HSHD_NUM = h.HSHD_NUM
                WHERE h.HH_SIZE NOT IN ('null','') AND h.HH_SIZE IS NOT NULL
                GROUP BY h.HH_SIZE ORDER BY h.HH_SIZE
            """,
            "children": """
                SELECT
                    CASE WHEN h.CHILDREN = 'null' OR h.CHILDREN IS NULL THEN 'Unknown'
                         WHEN TRY_CAST(h.CHILDREN AS FLOAT) > 0 THEN 'Has Children'
                         ELSE 'No Children' END as children_status,
                    AVG(CAST(t.SPEND AS DECIMAL(10,2))) as avg_spend,
                    COUNT(DISTINCT t.HSHD_NUM) as hh_count
                FROM transactions t JOIN households h ON t.HSHD_NUM = h.HSHD_NUM
                GROUP BY
                    CASE WHEN h.CHILDREN = 'null' OR h.CHILDREN IS NULL THEN 'Unknown'
                         WHEN TRY_CAST(h.CHILDREN AS FLOAT) > 0 THEN 'Has Children'
                         ELSE 'No Children' END
            """,
            "region": """
                SELECT STORE_R,
                       SUM(CAST(SPEND AS DECIMAL(10,2))) as total_spend,
                       COUNT(DISTINCT HSHD_NUM) as unique_hh
                FROM transactions
                WHERE STORE_R NOT IN ('null','') AND STORE_R IS NOT NULL
                GROUP BY STORE_R ORDER BY total_spend DESC
            """,
            "weekly": """
                SELECT CAST(WEEK_NUM AS INT) as WEEK_NUM,
                       CAST(YEAR AS INT) as YEAR,
                       SUM(CAST(SPEND AS DECIMAL(10,2))) as spend
                FROM transactions
                GROUP BY WEEK_NUM, YEAR
                ORDER BY YEAR, CAST(WEEK_NUM AS INT)
            """,
            "dept_year": """
                SELECT p.DEPARTMENT, CAST(t.YEAR AS INT) as YEAR,
                       SUM(CAST(t.SPEND AS DECIMAL(10,2))) as total_spend
                FROM transactions t JOIN products p ON t.PRODUCT_NUM = p.PRODUCT_NUM
                WHERE t.YEAR NOT IN ('null','') AND t.YEAR IS NOT NULL
                GROUP BY p.DEPARTMENT, t.YEAR ORDER BY t.YEAR, total_spend DESC
            """,
            "basket": """
                SELECT TOP 10 c1.COMMODITY as item_1, c2.COMMODITY as item_2,
                    COUNT(*) as times_bought_together
                FROM (
                    SELECT DISTINCT t.BASKET_NUM, p.COMMODITY
                    FROM transactions t JOIN products p ON t.PRODUCT_NUM = p.PRODUCT_NUM
                    WHERE p.COMMODITY NOT IN ('null','') AND p.COMMODITY IS NOT NULL
                ) c1
                JOIN (
                    SELECT DISTINCT t.BASKET_NUM, p.COMMODITY
                    FROM transactions t JOIN products p ON t.PRODUCT_NUM = p.PRODUCT_NUM
                    WHERE p.COMMODITY NOT IN ('null','') AND p.COMMODITY IS NOT NULL
                ) c2 ON c1.BASKET_NUM = c2.BASKET_NUM AND c1.COMMODITY < c2.COMMODITY
                GROUP BY c1.COMMODITY, c2.COMMODITY
                ORDER BY times_bought_together DESC
            """,
            "seasonal": """
                SELECT
                    CASE WHEN CAST(WEEK_NUM AS INT) BETWEEN 1  AND 13 THEN 'Spring'
                         WHEN CAST(WEEK_NUM AS INT) BETWEEN 14 AND 26 THEN 'Summer'
                         WHEN CAST(WEEK_NUM AS INT) BETWEEN 27 AND 39 THEN 'Fall'
                         ELSE 'Winter' END as season,
                    SUM(CAST(SPEND AS DECIMAL(10,2))) as total_spend,
                    AVG(CAST(SPEND AS DECIMAL(10,2))) as avg_spend,
                    COUNT(DISTINCT HSHD_NUM) as unique_hh
                FROM transactions
                WHERE WEEK_NUM NOT IN ('null','') AND WEEK_NUM IS NOT NULL
                GROUP BY
                    CASE WHEN CAST(WEEK_NUM AS INT) BETWEEN 1  AND 13 THEN 'Spring'
                         WHEN CAST(WEEK_NUM AS INT) BETWEEN 14 AND 26 THEN 'Summer'
                         WHEN CAST(WEEK_NUM AS INT) BETWEEN 27 AND 39 THEN 'Fall'
                         ELSE 'Winter' END
            """,
            "seasonal_commodity": """
                SELECT TOP 20
                    CASE WHEN CAST(t.WEEK_NUM AS INT) BETWEEN 1  AND 13 THEN 'Spring'
                         WHEN CAST(t.WEEK_NUM AS INT) BETWEEN 14 AND 26 THEN 'Summer'
                         WHEN CAST(t.WEEK_NUM AS INT) BETWEEN 27 AND 39 THEN 'Fall'
                         ELSE 'Winter' END as season,
                    p.DEPARTMENT,
                    SUM(CAST(t.SPEND AS DECIMAL(10,2))) as total_spend
                FROM transactions t JOIN products p ON t.PRODUCT_NUM = p.PRODUCT_NUM
                WHERE t.WEEK_NUM NOT IN ('null','') AND t.WEEK_NUM IS NOT NULL
                GROUP BY
                    CASE WHEN CAST(t.WEEK_NUM AS INT) BETWEEN 1  AND 13 THEN 'Spring'
                         WHEN CAST(t.WEEK_NUM AS INT) BETWEEN 14 AND 26 THEN 'Summer'
                         WHEN CAST(t.WEEK_NUM AS INT) BETWEEN 27 AND 39 THEN 'Fall'
                         ELSE 'Winter' END,
                    p.DEPARTMENT
                ORDER BY season, total_spend DESC
            """,
            "brand": """
                SELECT p.BRAND_TY,
                       SUM(CAST(t.SPEND AS DECIMAL(10,2))) as total_spend,
                       COUNT(*) as transaction_count
                FROM transactions t JOIN products p ON t.PRODUCT_NUM = p.PRODUCT_NUM
                WHERE p.BRAND_TY NOT IN ('null','') AND p.BRAND_TY IS NOT NULL
                GROUP BY p.BRAND_TY
            """,
            "organic": """
                SELECT p.NATURAL_ORGANIC_FLAG,
                       SUM(CAST(t.SPEND AS DECIMAL(10,2))) as total_spend,
                       COUNT(DISTINCT t.HSHD_NUM) as buyers
                FROM transactions t JOIN products p ON t.PRODUCT_NUM = p.PRODUCT_NUM
                WHERE p.NATURAL_ORGANIC_FLAG NOT IN ('null','') AND p.NATURAL_ORGANIC_FLAG IS NOT NULL
                GROUP BY p.NATURAL_ORGANIC_FLAG
            """,
            "brand_income": """
                SELECT h.INCOME_RANGE, p.BRAND_TY,
                       SUM(CAST(t.SPEND AS DECIMAL(10,2))) as total_spend
                FROM transactions t
                JOIN products p ON t.PRODUCT_NUM = p.PRODUCT_NUM
                JOIN households h ON t.HSHD_NUM = h.HSHD_NUM
                WHERE p.BRAND_TY NOT IN ('null','') AND p.BRAND_TY IS NOT NULL
                  AND h.INCOME_RANGE NOT IN ('null','') AND h.INCOME_RANGE IS NOT NULL
                GROUP BY h.INCOME_RANGE, p.BRAND_TY ORDER BY h.INCOME_RANGE, p.BRAND_TY
            """,
            "dept": """
                SELECT p.DEPARTMENT,
                       SUM(CAST(t.SPEND AS DECIMAL(10,2))) as total_spend,
                       COUNT(DISTINCT t.HSHD_NUM) as unique_households
                FROM transactions t JOIN products p ON t.PRODUCT_NUM = p.PRODUCT_NUM
                GROUP BY p.DEPARTMENT ORDER BY total_spend DESC
            """,
            "commodity": """
                SELECT TOP 10 p.COMMODITY,
                       SUM(CAST(t.SPEND AS DECIMAL(10,2))) as total_spend
                FROM transactions t JOIN products p ON t.PRODUCT_NUM = p.PRODUCT_NUM
                GROUP BY p.COMMODITY ORDER BY total_spend DESC
            """,
        }

        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
            fmap = {ex.submit(_q, sql): name for name, sql in SQLS.items()}
            for fut in concurrent.futures.as_completed(fmap):
                name = fmap[fut]
                try:
                    results[name] = fut.result()
                except Exception:
                    results[name] = pd.DataFrame()

        income             = results.get("income",             pd.DataFrame())
        hhsize             = results.get("hhsize",             pd.DataFrame())
        children           = results.get("children",           pd.DataFrame())
        region             = results.get("region",             pd.DataFrame())
        weekly             = results.get("weekly",             pd.DataFrame())
        dept_year          = results.get("dept_year",          pd.DataFrame())
        basket             = results.get("basket",             pd.DataFrame())
        seasonal           = results.get("seasonal",           pd.DataFrame())
        seasonal_commodity = results.get("seasonal_commodity", pd.DataFrame())
        brand              = results.get("brand",              pd.DataFrame())
        organic            = results.get("organic",            pd.DataFrame())
        brand_income       = results.get("brand_income",       pd.DataFrame())
        dept               = results.get("dept",               pd.DataFrame())
        commodity          = results.get("commodity",          pd.DataFrame())

        # ── ML — only run if models are already trained (never block dashboard) ─
        top_clv, churn_counts, avg_clv, high_risk_count = [], {}, 0, 0
        if (os.path.exists("models/clv_model.pkl") and
                os.path.exists("models/churn_model.pkl")):
            try:
                from ml_models import get_all_predictions
                preds = get_all_predictions()
                top_clv = preds.nlargest(10, 'clv_score')[
                    ['HSHD_NUM','clv_score','churn_prob','risk_segment','frequency','recency']
                ].to_dict("records")
                churn_counts    = preds['risk_segment'].value_counts().to_dict()
                avg_clv         = round(preds['clv_score'].mean(), 2)
                high_risk_count = int((preds['risk_segment'] == 'High Risk').sum())
            except Exception:
                pass

        _dashboard_cache["data"] = {
            "ts": time.time(),
            "income": income, "hhsize": hhsize, "children": children,
            "region": region, "weekly": weekly, "dept_year": dept_year,
            "basket": basket, "seasonal": seasonal,
            "seasonal_commodity": seasonal_commodity, "brand": brand,
            "organic": organic, "brand_income": brand_income,
            "dept": dept, "commodity": commodity,
            "top_clv": top_clv, "churn_counts": churn_counts,
            "avg_clv": avg_clv, "high_risk_count": high_risk_count,
        }

    return render_template("dashboard.html",
        dept=dept.to_dict("records"),
        weekly=weekly.to_dict("records"),
        income=income.to_dict("records"),
        brand=brand.to_dict("records"),
        organic=organic.to_dict("records"),
        commodity=commodity.to_dict("records"),
        hhsize=hhsize.to_dict("records"),
        children=children.to_dict("records"),
        region=region.to_dict("records"),
        dept_year=dept_year.to_dict("records"),
        basket=basket.to_dict("records"),
        seasonal=seasonal.to_dict("records"),
        seasonal_commodity=seasonal_commodity.to_dict("records"),
        brand_income=brand_income.to_dict("records"),
        top_clv=top_clv,
        churn_counts=churn_counts,
        avg_clv=avg_clv,
        high_risk_count=high_risk_count
    )
# ── ML Results (Req 7 & 8) ────────────────────────────────────────────────────

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