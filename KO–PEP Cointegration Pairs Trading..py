#!/usr/bin/env python
# coding: utf-8

# In[110]:


import yfinance as yf
import numpy as np
import pandas as pd
import statsmodels

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import seaborn




import statsmodels.tsa.stattools as ts

import warnings
warnings.filterwarnings("ignore")


from datetime import datetime




# In[ ]:





# In[111]:


x=yf.download("KO",start="2019-01-01", end="2022-12-31",auto_adjust=True)

y= yf.download("PEP", start="2019-01-01", end="2022-12-31",auto_adjust=True)


# In[112]:


x.index = pd.to_datetime(x.index)
y.index = pd.to_datetime(y.index)




y.isna().sum()
x.isna().sum()
x.duplicated().value_counts()
y.duplicated().value_counts()


# In[113]:


duplicados_x = x.index.duplicated().sum()
duplicados_y = y.index.duplicated().sum()

print(f"Duplicados en x (KO): {duplicados_x}")
print(f"Duplicados en y (PEP): {duplicados_y}")


# In[114]:


print(type(x.columns ))
print(type(y.columns ))


# In[115]:


if isinstance(x.columns, pd.MultiIndex):
    x.columns = x.columns.droplevel(1)
    
if isinstance(y.columns, pd.MultiIndex):
    y.columns = y.columns.droplevel(1)


# In[116]:


print(type(x.columns ))
print(type(y.columns ))


# In[117]:


df = pd.DataFrame({'KO': x['Close'], 'PEP': y['Close']})
Co=df[["KO","PEP"]]
Co.plot(figsize=(10,5))
plt.show()


# In[118]:


import statsmodels.api as sm


# In[119]:


#Calculate de hedge ratio

beta=sm.OLS(df.KO.iloc[:90],df.PEP.iloc[:90])
beta=beta.fit().params[0]
print(f"Hedge Ratio (beta) = {beta:.4f}")

plt.scatter(df.KO,df.PEP)
plt.xlabel("Coke")
plt.ylabel("Pepsi")
plt.show()


# In[120]:


df["spread"]=df.KO - beta * df.PEP
df["spread"].plot(figsize=(10,5))
plt.ylabel(f"KO - {beta:.2f}*PEP")
plt.show()


# In[121]:


adf= ts.adfuller(df["spread"])

print("ADF Test Stats", adf[0] )
print("ADF Critical Values",adf[4])


# In[122]:


def zscore(series, window=90):
    mean=series.rolling(window, min_periods=window).mean()
    std=series.rolling(window, min_periods=window).std()
    z= (series - mean)/std
    return z.replace([np.inf,-np.inf],np.nan)

df["z_score"]=zscore(df["spread"],window=90)


# In[ ]:





# In[123]:


plt.figure(figsize=(10,5))
df["z_score"].plot()
plt.axhline(0,color="black")
plt.axhline(2.6,color="red", linestyle ="--")
plt.axhline(-2.6,color="blue",linestyle="--")
plt.title('Z-Score del Spread KO–PEP (window=90)')
plt.show()


# In[124]:


def rolling_thresholds(z_score,window=250,upper_q=0.95,lower_q=0.05):
    upper=z_score.rolling(window,min_periods=window).quantile(upper_q)
    lower=z_score.rolling(window,min_periods=window).quantile(lower_q)
    return upper,lower
df['z_upper'], df['z_lower'] = rolling_thresholds(df['z_score'], window=250, upper_q=0.95, lower_q=0.05)
df['exit_band'] = (0.20 * (df['z_upper'] - df['z_lower'])).clip(lower=0.3)


# In[125]:


df['long_signal']  = (df['z_score'] < df['z_lower']).astype(int)
df['short_signal'] = (df['z_score'] > df['z_upper']).astype(int)
df['exit_signal']  = (df['z_score'].abs() < df['exit_band']).astype(int)


# In[126]:


desired = pd.Series(np.nan, index=df.index)
desired.loc[df['long_signal']  == 1] =  1
desired.loc[df['short_signal'] == 1] = -1


# In[127]:


desired.loc[df['exit_signal'] == 1] = 0


# In[128]:


df['position'] = desired.ffill().fillna(0)


# In[129]:


df['r_KO']  = df['KO'].pct_change()
df['r_PEP'] = df['PEP'].pct_change()


# In[130]:


w_KO  = df['position'].shift(1).fillna(0)       
w_PEP = (-beta * df['position']).shift(1).fillna(0)  


# In[131]:


df['strategy_ret'] = w_KO * df['r_KO'] + w_PEP * df['r_PEP']
df['cum_ret'] = (1 + df['strategy_ret'].fillna(0)).cumprod()


# In[132]:


fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
df[['KO','PEP']].plot(ax=axes[0], title='KO vs PEP (Close)')
axes[0].grid(True)

df['z_score'].plot(ax=axes[1], color='black', label='z-score')
df['z_upper'].plot(ax=axes[1], linestyle='--', label='Upper (roll)')
df['z_lower'].plot(ax=axes[1], linestyle='--', label='Lower (roll)')
(df['exit_band']).plot(ax=axes[1], linestyle=':', label='Exit +band')
(-df['exit_band']).plot(ax=axes[1], linestyle=':', label='Exit -band')
axes[1].axhline(0, color='gray', linewidth=1)
axes[1].legend(); axes[1].grid(True); axes[1].set_title('Z-score y Umbrales Dinámicos')

df['cum_ret'].plot(ax=axes[2], label='Estrategia KO−β·PEP (cum.)')
axes[2].legend(); axes[2].grid(True); axes[2].set_title('Retorno Acumulado (sin comisiones)')

plt.tight_layout()
plt.show()


# In[133]:


df.tail(20)


# In[134]:


cols = ['z_score','z_upper','z_lower','long_signal','short_signal','exit_signal']
print("NaNs:\n", df[cols].isna().sum(), "\n")

ready = df[['z_score','z_upper','z_lower']].notnull().all(axis=1)
print("Primer día listo:", df.index[ready.argmax()] if ready.any() else "ninguno")
print("¿Hay long? ", (df['long_signal']==1).any())
print("¿Hay short?',", (df['short_signal']==1).any())


# In[135]:


plt.savefig("zscore_thresholds.png", dpi=150, bbox_inches="tight")
plt.savefig("cumulative_returns.png", dpi=150, bbox_inches="tight")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:






# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




