# Credit Card Fraud Detection
# Author: Vinod M | mopurivinod6788@gmail.com

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# =====================
# 1. CREATE SAMPLE DATA
# =====================
np.random.seed(42)
n = 10000

data = {
    'transaction_id': range(1, n+1),
    'amount': np.round(np.random.exponential(100, n), 2),
    'age': np.random.randint(18, 70, n),
    'num_transactions_today': np.random.randint(1, 20, n),
    'time_since_last_transaction': np.round(np.random.exponential(5, n), 2),
    'distance_from_home': np.round(np.random.exponential(20, n), 2),
    'online_order': np.random.choice([0, 1], n, p=[0.4, 0.6]),
    'used_pin': np.random.choice([0, 1], n, p=[0.3, 0.7]),
    'fraud': np.random.choice([0, 1], n, p=[0.97, 0.03])  # imbalanced!
}

df = pd.DataFrame(data)

# =====================
# 2. DATA OVERVIEW
# =====================
print("=== DATA OVERVIEW ===")
print(f"Total Transactions: {len(df)}")
print(f"Fraud Cases: {df['fraud'].sum()} ({df['fraud'].mean()*100:.2f}%)")
print(f"Legit Cases: {(df['fraud']==0).sum()}")
print(f"\nMissing Values:\n{df.isnull().sum()}")

# =====================
# 3. VISUALIZATIONS
# =====================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Credit Card Fraud Detection - Vinod M', fontsize=16, fontweight='bold')

# Chart 1: Class Imbalance
fraud_counts = df['fraud'].value_counts()
axes[0, 0].bar(['Legitimate', 'Fraud'], fraud_counts.values, color=['#10B981', '#EF4444'])
axes[0, 0].set_title('Class Distribution (Imbalanced)')
axes[0, 0].set_ylabel('Number of Transactions')
for i, v in enumerate(fraud_counts.values):
    axes[0, 0].text(i, v + 50, str(v), ha='center', fontweight='bold')

# Chart 2: Transaction Amount Distribution
axes[0, 1].hist(df[df['fraud']==0]['amount'], bins=50, alpha=0.7, label='Legitimate', color='#10B981')
axes[0, 1].hist(df[df['fraud']==1]['amount'], bins=50, alpha=0.7, label='Fraud', color='#EF4444')
axes[0, 1].set_title('Transaction Amount Distribution')
axes[0, 1].set_xlabel('Amount (₹)')
axes[0, 1].set_ylabel('Count')
axes[0, 1].legend()

# Chart 3: Distance from Home
axes[1, 0].hist(df[df['fraud']==0]['distance_from_home'], bins=50, alpha=0.7, label='Legitimate', color='#2563EB')
axes[1, 0].hist(df[df['fraud']==1]['distance_from_home'], bins=50, alpha=0.7, label='Fraud', color='#F59E0B')
axes[1, 0].set_title('Distance from Home')
axes[1, 0].set_xlabel('Distance (km)')
axes[1, 0].legend()

# Chart 4: Online vs In-person Fraud
online_fraud = df.groupby('online_order')['fraud'].mean() * 100
axes[1, 1].bar(['In-Person', 'Online'], online_fraud.values, color=['#2563EB', '#EF4444'])
axes[1, 1].set_title('Fraud Rate: Online vs In-Person')
axes[1, 1].set_ylabel('Fraud Rate (%)')

plt.tight_layout()
plt.savefig('fraud_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Chart saved as fraud_analysis.png")

# =====================
# 4. PREPARE DATA
# =====================
features = ['amount', 'age', 'num_transactions_today',
            'time_since_last_transaction', 'distance_from_home',
            'online_order', 'used_pin']
X = df[features]
y = df['fraud']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test =