import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device('cpu')

# 1. ዴታውን ማንበብ
df = pd.read_csv('ethiopia_real_estate_market_2026_v2.csv')

# የቤቱን ዕድሜ (Age) ማስላት
df['Property_Age'] = 2026 - df['Year_Built']

# --- 2. ቪዥዋል (Visualizations) ክፍሎች ---
df['Price_in_Millions'] = df['Price_ETB'] / 1_000_000
avg_price = df.groupby('Location')['Price_in_Millions'].mean()

plt.figure(figsize=(10, 5))
avg_price.plot(kind='bar', color='skyblue')
plt.title('Average Price by Location (in Millions ETB)')
plt.ylabel('Price in Millions ETB')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

location_market_share = df.groupby('Location')['Price_ETB'].sum()

plt.figure(figsize=(9, 9))
plt.pie(
    location_market_share, 
    labels=location_market_share.index, 
    autopct='%1.1f%%', 
    startangle=140,
    colors=sns.color_palette('pastel'),
    shadow=True
)
plt.title('Real Estate Market Value Share by Location (%)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

real_estate_market = df.groupby(['Location','Bedrooms'])['Price_ETB'].sum().reset_index()
plt.figure(figsize=(14, 6))
sns.barplot(data=real_estate_market, x='Location', y='Price_ETB', hue='Bedrooms')
plt.title('Total Price by Location and Bedrooms')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Categorical መረጃዎችን ወደ ቁጥር መቀየር (One-Hot Encoding)
df_encoded = pd.get_dummies(df, columns=['Location', 'Property_Type'], drop_first=True)

# 4. Features (X) እና Target (y) መለየት
X = df_encoded.drop(columns=['Property_ID', 'Price_ETB', 'Price_USD', 'Year_Built', 'Price_in_Millions'])
y = df_encoded['Price_ETB']

# 5. መረጃውን ለስልጠና (80%) እና ለፈተና (20%) መክፈል
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. ዳታውን እና ታርጌቱን ማስተካከል (StandardScaler - ለ X እና y)
scaler_x = StandardScaler()
X_train_scaled = scaler_x.fit_transform(X_train)
X_test_scaled = scaler_x.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

# ወደ PyTorch Tensors መቀየር (ማስጠንቀቂያውን ለማስወገድ .copy() እንጠቀማለን)
X_train_tensor = torch.FloatTensor(X_train_scaled.copy())
y_train_tensor = torch.FloatTensor(y_train_scaled.copy())
X_test_tensor = torch.FloatTensor(X_test_scaled.copy())
y_test_tensor = torch.FloatTensor(y_test_scaled.copy())

# 7. የ PyTorch Neural Network ሞዴል መዋቅር መገንባት
class RealEstateNN(nn.Module):
    def __init__(self, input_dim):
        super(RealEstateNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(32, 1)
        
    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.output(x)
        return x

input_dim = X_train.shape[1]
model = RealEstateNN(input_dim)

# Loss Function እና Optimizer መምረጥ
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 8. ሞዴሉን ማሰልጠን (Training Loop)
epochs = 200
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    # ፊት ለፊት ማለፍ (Forward pass)
    predictions = model(X_train_tensor)
    loss = criterion(predictions, y_train_tensor)
    
    # ኋላቀር ማለፍ እና ክብደቶችን ማስተካከል (Backward pass & optimization)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():,.4f}')

# 9. ትንበያ መስራት እና ውጤቱን ወደ ትክክለኛው የብር መጠን መመለስ
model.eval()
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor)
    y_pred_scaled = y_pred_tensor.numpy()
    
# የተሰላውን ስኬል ወደ መጀመሪያው የብር መጠን መመለስ (Inverse Transform)
y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
y_test_actual = scaler_y.inverse_transform(y_test_tensor.numpy()).flatten()

mse = mean_squared_error(y_test_actual, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_actual, y_pred)

print(f"\n--- የ PyTorch Neural Network ሞዴል የተስተካከለ ውጤት ---")
print(f"Mean Squared Error (MSE): {mse:,.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:,.2f} ETB")
print(f"Model R2 Score: {r2:.4f} ({r2 * 100:.2f}% ትክክለኛነት)\n")

# 10. እውነተኛ ዋጋ vs የትንበያ ዋጋ ግራፍ
plt.figure(figsize=(8, 8))
plt.scatter(y_test_actual / 1_000_000, y_pred / 1_000_000, color='purple', alpha=0.5)
plt.plot([y_test_actual.min()/1_000_000, y_test_actual.max()/1_000_000], 
         [y_test_actual.min()/1_000_000, y_test_actual.max()/1_000_000], 
         color='red', linestyle='--', linewidth=2, label='Perfect Prediction Line')
plt.title('Actual Prices vs Predicted Prices (PyTorch NN)', fontsize=14, fontweight='bold')
plt.xlabel('Actual Price (Millions ETB)')
plt.ylabel('Predicted Price (Millions ETB)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()