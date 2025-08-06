# Predictive Analysis for Demand Forecast in E-Commerce Supply Chains

![Project Header](head.pdf)

## Project Overview
A comprehensive predictive analytics framework for e-commerce supply chain optimization, built on the Olist Brazilian E-Commerce Dataset. This project demonstrates end-to-end data science workflow from raw transactional data to actionable business insights.

## Key Achievements
- **99.5%** data quality coverage
- **91.3%** overall validation score  
- **R² = 0.967** demand forecasting accuracy
- **24.3%** inventory turnover improvement
- **22.7%** delivery time reduction
- **18.6%** delivery cost reduction

## Framework Architecture

![Framework](figures/Framework.pdf)

The framework demonstrates the multi-stage analytical approach from data infrastructure to implementation validation, integrating customer behavior analysis, product lifecycle modeling, demand forecasting, and operational optimization into a unified decision support system.

### Data Infrastructure Layer
- ETL pipeline with comprehensive validation
- PostgreSQL database with 3NF schema
- 16 processed tables across 5 functional categories

### Analytics & Insights Layer  
- Customer RFM segmentation (96,683 profiles)
- Product lifecycle classification (88.5% Growth, 8.2% Maturity, 2.4% Decline)
- Holiday sensitivity analysis (35% sales increase during holidays)

### Predictive Modeling Layer
- Ensemble forecasting: ARIMA, XGBoost, LightGBM, LSTM, GRU, Transformer
- Time series feature engineering with lag features and seasonal components
- Cross-validation ensuring model robustness

### Optimization & Decision Layer
- Warehouse capacity planning with discrete event simulation
- Four-dimensional analysis (Seller-Product-Geography-Time)
- Cost-benefit analysis with Monte Carlo simulation

### Implementation & Validation Layer
- Risk assessment and mitigation strategies
- Recommendation generation with feasibility scoring
- Historical backtesting and A/B testing design

## Dataset Information
- **Source**: Olist Brazilian E-Commerce Dataset
- **Period**: January 2016 - October 2018 (33 months)
- **Scale**: 99,441 orders, 112,650 order items, 3,095 sellers
- **Coverage**: 27 Brazilian states, 71 product categories
- **Files**: 9 primary CSV files with comprehensive e-commerce data

## Project Structure
```
Olist_Ecommerce_Analysis_Project/
├── 1_data_preparation/          # Data Infrastructure and ETL Pipeline
│   ├── 0.data_exploration.py    # Initial data exploration and quality assessment
│   ├── 1.base_*.py              # Base table processing (customers, orders, products, etc.)
│   ├── 2.derived_*.py           # Derived tables (customer profiles, product lifecycle, etc.)
│   ├── 3.dim_*.py               # Dimension tables (date, location)
│   ├── 4.check_tables.py        # Data validation and quality checks
│   ├── SQL/                     # Database schema and setup scripts
│   └── run_all.sh               # Complete ETL pipeline execution
├── 2_eda_update/                # Exploratory Data Analysis
│   ├── eda_timeseries_aggregation.py      # Time series analysis
│   ├── product_performance_overview.py    # Product performance metrics
│   ├── sales_distribution_*.py            # Sales analysis by dimensions
│   └── weekend_holiday_*.py              # Holiday and weekend analysis
├── 3_customer_behavior/         # Customer Behavior Analysis
│   ├── rfm_logistics_segmentation.py     # RFM customer segmentation
│   ├── customer_lifecycle_classification.py # Customer lifecycle analysis
│   ├── product_preference_segmentation.py # Product preference analysis
│   ├── purchase_funnel_dropout_analysis.py # Purchase funnel analysis
│   └── final_customer_persona_table.py   # Customer persona generation
├── 4_product_warehouse_analysis/ # Product and Warehouse Analysis
│   ├── product_lifecycle_classification.py # Product lifecycle analysis
│   ├── product_sales_curve_analysis.py    # Sales curve modeling
│   ├── inventory_efficiency_analysis.py   # Inventory optimization
│   ├── stock_risk_detection.py           # Risk assessment
│   ├── warehouse_simulation.py           # Warehouse simulation models
│   └── inventory_policy_recommendation.py # Policy recommendations
├── 5_seller_analysis_and_four_d_analysis/ # Seller Analysis Framework
│   ├── seller_lifecycle_product_strategy_analysis.py # Seller lifecycle analysis
│   ├── seller_fulfillment_complexity_analysis.py     # Fulfillment complexity
│   ├── seller_warehouse_demand_analysis.py          # Warehouse demand analysis
│   ├── regional_fulfillment_load_projection.py      # Regional load projection
│   └── seller_product_geography_time_analysis.py    # 4D analysis framework
├── 6_forecasts/                 # Demand Forecasting Models
│   ├── 1_time_series_feature_engineering.py # Feature engineering
│   ├── 2.1_statistical_forecasting.py      # Statistical models (ARIMA, etc.)
│   ├── 2.2_ml_forecasting.py              # Machine learning models
│   ├── 2.3_dl_forecasting.py              # Deep learning models (LSTM, GRU)
│   ├── 2.4_model_ensemble.py              # Ensemble forecasting
│   ├── 3.1_basic_capacity_calculation.py  # Capacity planning
│   └── multi_level_demand_forecasting.py  # Multi-level forecasting
├── 7_accurate_recommendations/  # Financial Analysis and Recommendations
│   ├── 1.1_cost_benefit_analysis_engine.py # Cost-benefit analysis
│   ├── 2.1_risk_assessment_mitigation.py  # Risk assessment
│   ├── 3.1_precision_recommendation_generator.py # Recommendation generation
│   ├── 4.1_recommendation_validation_optimization.py # Validation framework
│   └── run_week7_main.py                  # Main execution script
├── data/                        # Raw and processed datasets
├── requirements.txt              
└── README.md                    
```

## Technology Stack
- **Programming**: Python 3.8+
- **Data Processing**: Pandas, NumPy, SQLAlchemy
- **Database**: PostgreSQL
- **Machine Learning**: Scikit-learn, XGBoost, LightGBM
- **Deep Learning**: TensorFlow/Keras (LSTM, GRU, Transformer)
- **Time Series**: Statsmodels, Prophet
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Documentation**: LaTeX (Tau class template)

## Installation and Setup

### Prerequisites
- Python 3.8+
- PostgreSQL 12+
- Required Python packages (see requirements.txt)

### Quick Start
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up PostgreSQL database
4. Run data preparation: `python 1_data_preparation/run_all.sh`
5. Execute analysis pipeline: Follow weekly progression (1-7)

### Running Individual Components
- Data preparation: `cd 1_data_preparation && python run_all.sh`
- EDA: `cd 2_eda_update && python eda_timeseries_aggregation.py`
- Customer analysis: `cd 3_customer_behavior && python rfm_logistics_segmentation.py`
- Forecasting: `cd 6_forecasts && python 2.4_model_ensemble.py`

## Deployment Requirements

### System Requirements
- **CPU**: Minimum 4 cores, recommended 8+ cores for large datasets
- **RAM**: Minimum 8GB, recommended 16GB+ for deep learning models
- **Storage**: Minimum 10GB free space for datasets and models
- **GPU**: Optional but recommended for deep learning (TensorFlow GPU support)

### Software Dependencies
- **Operating System**: Linux (Ubuntu 18.04+), macOS (10.15+), or Windows 10+
- **Python**: 3.8 or higher
- **PostgreSQL**: 12.0 or higher
- **Git**: For version control

### Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Unix/MacOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up PostgreSQL database
# Create database and configure connection settings
```

### Configuration
- Update database connection settings in configuration files
- Set up environment variables for sensitive data
- Configure logging and output directories

## Key Findings and Results

### Customer Insights
- 96,683 customer profiles with 41 behavioral features
- RFM segmentation with 0.72 silhouette score
- Holiday-sensitive products identified (health_beauty: +35%, watches_gifts: +25%)

### Product Analysis  
- Product lifecycle classification with 88.5% in Growth stage
- Inventory risk flags for 27.2% high-volatility products
- 24.3% inventory turnover improvement achieved

### Forecasting Performance
- Ensemble model: R² = 0.967, MAE = 12.1, MAPE = 3.9%
- Best individual model: XGBoost (R² = 0.999998)
- Cross-validation ensures model robustness

### Financial Impact
- ROI: 244.9%, NPV: $1,003,503,228
- Payback period: 4.48 years
- 100% probability of positive NPV across scenarios

## Documentation
- **White Paper**: Comprehensive LaTeX document with technical details
- **Code Examples**: Representative implementations in Appendix
- **Visualizations**: Framework diagrams and analysis charts
- **Results**: Detailed findings and recommendations

## Reports Generated
- Data quality assessment reports
- Customer segmentation analysis
- Product lifecycle classification
- Demand forecasting models
- Warehouse optimization recommendations
- Financial analysis and risk assessment

## Acknowledgments
- Olist Brazilian E-Commerce Dataset
- Tau LaTeX class template
- Open source community contributions
