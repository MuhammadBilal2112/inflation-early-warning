"""
==============================================================================
STEP 4: EXPLORATORY DATA ANALYSIS (EDA)
==============================================================================

Generates ~13 publication-quality figures + 3 tables for Chapter 3.
All saved to outputs/figures/ and outputs/tables/

REQUIRES: Steps 2-3 completed
RUN: python3 Step04_Exploratory_Analysis.py
TIME: 1-2 minutes
==============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os, warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({'figure.figsize':(12,6),'figure.dpi':150,'font.size':11,
    'font.family':'sans-serif','axes.titlesize':13,'axes.labelsize':11,
    'legend.fontsize':9,'figure.facecolor':'white','axes.facecolor':'white',
    'axes.grid':True,'grid.alpha':0.3,'axes.spines.top':False,'axes.spines.right':False})

C = {'blue':'#1F4E79','red':'#C62828','green':'#2E7D32','amber':'#E8A838',
     'purple':'#6A1B9A','teal':'#00838F','gray':'#616161','light_blue':'#42A5F5'}
GC = {'Advanced economies':'#1F4E79','EMDEs':'#C62828'}

BASE_DIR = "inflation_regimes_project"
PROC = os.path.join(BASE_DIR,"data","processed")
FDIR = os.path.join(BASE_DIR,"outputs","figures"); os.makedirs(FDIR,exist_ok=True)
TDIR = os.path.join(BASE_DIR,"outputs","tables"); os.makedirs(TDIR,exist_ok=True)

print("Loading data...")
monthly = pd.read_csv(os.path.join(PROC,'master_panel_monthly.csv'),parse_dates=['date'])
annual = pd.read_csv(os.path.join(PROC,'master_panel_annual.csv'))
print(f"  Monthly: {monthly.shape} | Annual: {annual.shape}")

fc=0
def save_fig(name):
    global fc; fc+=1
    plt.savefig(os.path.join(FDIR,f"{name}.png"),bbox_inches='tight',facecolor='white'); plt.close()
    print(f"  Fig {fc}: {name}.png")

print("\n"+"="*60+"\nGENERATING FIGURES\n"+"="*60)

# ===== FIG 1: Global inflation trends =====
gm = monthly.groupby('date')['hcpi_yoy'].median().reset_index()
ae = monthly[monthly['country_group']=='Advanced economies'].groupby('date')['hcpi_yoy'].median().reset_index()
em = monthly[monthly['country_group']=='EMDEs'].groupby('date')['hcpi_yoy'].median().reset_index()

fig,ax=plt.subplots(figsize=(14,6))
ax.plot(gm['date'],gm['hcpi_yoy'],color=C['gray'],lw=1.5,label='Global median',alpha=0.7)
ax.plot(ae['date'],ae['hcpi_yoy'],color=C['blue'],lw=1.5,label='Advanced economies',alpha=0.9)
ax.plot(em['date'],em['hcpi_yoy'],color=C['red'],lw=1.5,label='EMDEs',alpha=0.9)
for d,l,y in [('1973-10-01','Oil shock I',16),('1979-06-01','Oil shock II',14),
              ('1997-07-01','Asian crisis',6),('2008-09-01','GFC',6),
              ('2020-03-01','COVID',4),('2022-02-01','Ukraine',12)]:
    dt=pd.Timestamp(d); ax.axvline(x=dt,color='gray',ls='--',alpha=0.4,lw=0.8)
    ax.annotate(l,xy=(dt,y),fontsize=8,color='gray',ha='center',style='italic',
                bbox=dict(boxstyle='round,pad=0.2',fc='white',alpha=0.8,ec='none'))
ax.set_ylabel('Year-on-year inflation (%)'); ax.set_title('Figure 1: Global inflation trends, 1971-2025')
ax.set_ylim(-5,30); ax.legend(loc='upper right'); ax.set_xlim(pd.Timestamp('1971-01-01'),pd.Timestamp('2025-03-01'))
save_fig('fig01_global_inflation_trends')

# ===== FIG 2: Distributions by year =====
fig,axes=plt.subplots(2,3,figsize=(14,8))
for i,yr in enumerate([1975,1985,1995,2005,2015,2022]):
    ax=axes.flatten()[i]; data=annual[annual['year']==yr]['hcpi'].dropna().clip(-10,50)
    ax.hist(data,bins=40,color=C['blue'],alpha=0.7,edgecolor='white',lw=0.5)
    ax.axvline(x=data.median(),color=C['red'],ls='--',lw=1.5,label=f'Median: {data.median():.1f}%')
    ax.set_title(f'{yr}',fontweight='bold'); ax.set_xlabel('Inflation (%)'); ax.set_xlim(-10,50); ax.legend(fontsize=8)
plt.suptitle('Figure 2: Cross-country distribution of headline inflation',fontsize=13,fontweight='bold',y=1.02)
plt.tight_layout(); save_fig('fig02_inflation_distributions_by_year')

# ===== FIG 3: Components comparison =====
fig,ax=plt.subplots(figsize=(14,6))
for col,lab,clr in [('hcpi_yoy','Headline',C['blue']),('fcpi_yoy','Food',C['green']),
                     ('ecpi_yoy','Energy',C['red']),('ccpi_yoy','Core',C['purple'])]:
    s=monthly[monthly['date']>='2000-01-01'].groupby('date')[col].median().reset_index()
    ax.plot(s['date'],s[col],label=lab,color=clr,lw=1.3)
ax.axhline(y=0,color='black',lw=0.5,alpha=0.3); ax.set_ylim(-15,25)
ax.set_ylabel('YoY inflation (%, global median)'); ax.set_title('Figure 3: Inflation components diverge during shocks (2000-2025)')
ax.legend(loc='upper left'); save_fig('fig03_inflation_components_comparison')

# ===== FIG 4: Commodity prices and inflation =====
fig,(ax1,ax2)=plt.subplots(2,1,figsize=(14,8),sharex=True)
comm=monthly[['date','oil_price','food_commodity_index']].drop_duplicates('date').sort_values('date')
m=comm['date']>='1990-01-01'
ax1.plot(comm.loc[m,'date'],comm.loc[m,'oil_price'],color=C['red'],lw=1.2,label='Crude oil ($/bbl)')
ax1t=ax1.twinx(); ax1t.plot(comm.loc[m,'date'],comm.loc[m,'food_commodity_index'],color=C['green'],lw=1.2,label='Food index')
ax1.set_ylabel('Oil ($/bbl)',color=C['red']); ax1t.set_ylabel('Food index',color=C['green'])
ax1.set_title('Figure 4a: Commodity prices'); ax1.legend(loc='upper left',fontsize=9); ax1t.legend(loc='upper right',fontsize=9)
gm2=monthly.groupby('date')['hcpi_yoy'].median().reset_index(); m2=gm2['date']>='1990-01-01'
ax2.plot(gm2.loc[m2,'date'],gm2.loc[m2,'hcpi_yoy'],color=C['blue'],lw=1.2)
ax2.fill_between(gm2.loc[m2,'date'],0,gm2.loc[m2,'hcpi_yoy'],alpha=0.15,color=C['blue'])
ax2.set_ylabel('Global median inflation (%)'); ax2.set_title('Figure 4b: Inflation follows commodity shocks')
plt.tight_layout(); save_fig('fig04_commodity_prices_and_inflation')

# ===== FIG 5: AE vs EMDE =====
fig,axes=plt.subplots(1,2,figsize=(14,5))
for g,clr in GC.items():
    s=monthly[monthly['country_group']==g].groupby('date')['hcpi_yoy'].median().reset_index()
    m=s['date']>='1990-01-01'; axes[0].plot(s.loc[m,'date'],s.loc[m,'hcpi_yoy'],label=g,color=clr,lw=1.3)
axes[0].set_title('Figure 5a: Median inflation by group'); axes[0].set_ylabel('Inflation (%)'); axes[0].legend(); axes[0].set_ylim(-2,20)
rec=annual[(annual['year']>=2019)&(annual['year']<=2023)&annual['country_group'].notna()].copy()
bd=[]; 
for yr in [2019,2020,2021,2022,2023]:
    for g in ['Advanced economies','EMDEs']:
        for v in rec[(rec['year']==yr)&(rec['country_group']==g)]['hcpi'].dropna():
            if -10<v<60: bd.append({'year':yr,'group':g,'inflation':v})
bdf=pd.DataFrame(bd)
if not bdf.empty:
    sns.boxplot(data=bdf,x='year',y='inflation',hue='group',palette=GC,ax=axes[1],fliersize=2,linewidth=0.8)
    axes[1].set_title('Figure 5b: Dispersion 2019-2023'); axes[1].set_ylabel('Inflation (%)'); axes[1].legend(fontsize=8)
plt.tight_layout(); save_fig('fig05_ae_vs_emde_inflation')

# ===== FIG 6: By income group =====
fig,ax=plt.subplots(figsize=(12,5))
for ig,clr in zip(['High income','Upper middle income','Lower middle income','Low income'],
                   [C['blue'],C['teal'],C['amber'],C['red']]):
    s=monthly[monthly['income_group']==ig].groupby('date')['hcpi_yoy'].median().reset_index()
    m=s['date']>='2000-01-01'; ax.plot(s.loc[m,'date'],s.loc[m,'hcpi_yoy'],label=ig,color=clr,lw=1.2)
ax.set_title('Figure 6: Median inflation by income group (2000-2025)'); ax.set_ylabel('Inflation (%)'); ax.legend(loc='upper left'); ax.set_ylim(-3,20)
save_fig('fig06_inflation_by_income_group')

# ===== FIG 7: Correlations =====
fig,axes=plt.subplots(1,2,figsize=(14,5))
mr=monthly[monthly['date']>='2000-01-01']
cc=['hcpi_yoy','fcpi_yoy','ecpi_yoy','ccpi_yoy','ppi_yoy']; cl=['Headline','Food','Energy','Core','PPI']
cm=mr[cc].corr(); cm.index=cl; cm.columns=cl
sns.heatmap(cm,annot=True,fmt='.2f',cmap='RdBu_r',center=0,ax=axes[0],vmin=-0.5,vmax=1,square=True,linewidths=0.5)
axes[0].set_title('Fig 7a: Component correlations\n(monthly, 2000-2025)')
ar=annual[(annual['year']>=2000)&annual['country_group'].notna()]
fc2=['hcpi','debt_gdp','fiscal_balance','ext_debt_gdp','private_credit_gdp','sovereign_rating']
fl2=['Inflation','Debt/GDP','Fiscal bal.','Ext.debt/GDP','Priv.credit/GDP','Sov.rating']
cf=ar[fc2].corr(); cf.index=fl2; cf.columns=fl2
sns.heatmap(cf,annot=True,fmt='.2f',cmap='RdBu_r',center=0,ax=axes[1],vmin=-0.5,vmax=0.5,square=True,linewidths=0.5)
axes[1].set_title('Fig 7b: Inflation vs fiscal\n(annual, 2000-2024)')
plt.tight_layout(); save_fig('fig07_correlation_heatmaps')

# ===== FIG 8: COVID/Ukraine =====
fig,axes=plt.subplots(1,2,figsize=(14,5))
for cc2,(nm,clr) in {'USA':('United States',C['blue']),'GBR':('UK',C['red']),'TUR':('Turkey',C['amber']),
                      'BRA':('Brazil',C['green']),'IND':('India',C['purple']),'DEU':('Germany',C['teal'])}.items():
    d=monthly[(monthly['country_code']==cc2)&(monthly['date']>='2019-01-01')&(monthly['date']<='2025-01-01')]
    if not d.empty: axes[0].plot(d['date'],d['hcpi_yoy'],label=nm,color=clr,lw=1.2)
axes[0].axvline(x=pd.Timestamp('2020-03-01'),color='gray',ls='--',alpha=0.5)
axes[0].axvline(x=pd.Timestamp('2022-02-01'),color='gray',ls='--',alpha=0.5)
axes[0].annotate('COVID',xy=(pd.Timestamp('2020-03-01'),-2),fontsize=8,color='gray',ha='center')
axes[0].annotate('Ukraine',xy=(pd.Timestamp('2022-02-01'),-2),fontsize=8,color='gray',ha='center')
axes[0].set_title('Figure 8a: Selected countries 2019-2025'); axes[0].set_ylabel('Inflation (%)'); axes[0].legend(fontsize=8,ncol=2); axes[0].set_ylim(-5,80)

sd=monthly[monthly['hcpi_yoy'].notna()&(monthly['date']>='2015-01-01')].copy()
sb=sd.groupby('date').apply(lambda x:pd.Series({'pct5':(x['hcpi_yoy']>5).mean()*100,'pct10':(x['hcpi_yoy']>10).mean()*100})).reset_index()
axes[1].fill_between(sb['date'],0,sb['pct5'],alpha=0.3,color=C['amber'],label='> 5%')
axes[1].fill_between(sb['date'],0,sb['pct10'],alpha=0.5,color=C['red'],label='> 10%')
axes[1].set_title('Figure 8b: Share of countries with elevated inflation'); axes[1].set_ylabel('% of countries'); axes[1].legend(); axes[1].set_ylim(0,100)
plt.tight_layout(); save_fig('fig08_covid_ukraine_shock')

# ===== FIG 9: Fiscal scatters =====
fig,axes=plt.subplots(1,3,figsize=(15,4.5))
ra=annual[(annual['year']>=2000)&annual['hcpi'].notna()].copy(); ra['hc']=ra['hcpi'].clip(-5,40)
for i,(var,lab,xlim) in enumerate([('debt_gdp','Govt debt/GDP',(0,200)),('fiscal_balance','Fiscal balance',(-30,15)),('ext_debt_gdp','Ext.debt/GDP',(0,300))]):
    m=ra[var].notna(); axes[i].scatter(ra.loc[m,var],ra.loc[m,'hc'],alpha=0.1,s=8,color=[C['blue'],C['green'],C['red']][i])
    axes[i].set_xlabel(lab); axes[i].set_ylabel('Inflation (%)'); axes[i].set_xlim(xlim); axes[i].set_ylim(-5,40)
    axes[i].set_title(f'Figure 9{"abc"[i]}: {lab} vs inflation')
plt.suptitle('Figure 9: Fiscal indicators and inflation (annual, 2000-2024)',fontsize=12,fontweight='bold',y=1.02)
plt.tight_layout(); save_fig('fig09_fiscal_inflation_scatters')

# ===== FIG 10: Data coverage =====
kv=['hcpi_yoy','fcpi_yoy','ecpi_yoy','ccpi_yoy','ppi_yoy','oil_price','debt_gdp','fiscal_balance','ext_debt_gdp','sovereign_rating','private_credit_gdp']
kl=['Headline','Food','Energy','Core','PPI','Oil price','Debt/GDP','Fiscal bal.','Ext.debt','Sov.rating','Credit/GDP']
cm2=[]; 
for dc in [1970,1980,1990,2000,2010,2020]:
    row=[]; s=monthly[(monthly['year']>=dc)&(monthly['year']<dc+10)]
    for v in kv: row.append(s[v].notna().mean()*100 if v in s.columns else 0)
    cm2.append(row)
cdf=pd.DataFrame(cm2,index=[f'{d}s' for d in [1970,1980,1990,2000,2010,2020]],columns=kl)
fig,ax=plt.subplots(figsize=(12,4))
sns.heatmap(cdf,annot=True,fmt='.0f',cmap='YlGnBu',ax=ax,vmin=0,vmax=100,linewidths=0.5,cbar_kws={'label':'% non-missing'})
ax.set_title('Figure 10: Data coverage by decade'); ax.set_ylabel('Decade')
plt.tight_layout(); save_fig('fig10_data_coverage_heatmap')

# ===== FIG 11: Top inflation episodes =====
pk=annual[(annual['year']>=2020)&(annual['year']<=2024)].groupby(['country_code','country_name'])['hcpi'].max().reset_index().sort_values('hcpi',ascending=False).head(15)
fig,ax=plt.subplots(figsize=(10,6))
ax.barh(range(len(pk)),pk['hcpi'].values,color=C['red'],alpha=0.8)
ax.set_yticks(range(len(pk))); ax.set_yticklabels(pk['country_name'].values); ax.invert_yaxis()
ax.set_xlabel('Peak annual inflation (%)'); ax.set_title('Figure 11: Highest inflation rates, 2020-2024')
for i,v in enumerate(pk['hcpi'].values): ax.text(v+1,i,f'{v:.1f}%',va='center',fontsize=9)
plt.tight_layout(); save_fig('fig11_top_inflation_episodes')

# ===== FIG 12: Commodity volatility =====
fig,ax=plt.subplots(figsize=(14,5))
cv=monthly[['date','energy_index_6m_vol','food_commodity_index_6m_vol']].drop_duplicates('date').sort_values('date')
m=cv['date']>='1990-01-01'
ax.plot(cv.loc[m,'date'],cv.loc[m,'energy_index_6m_vol'],label='Energy volatility',color=C['red'],lw=1)
ax.plot(cv.loc[m,'date'],cv.loc[m,'food_commodity_index_6m_vol'],label='Food volatility',color=C['green'],lw=1)
ax.set_title('Figure 12: Commodity price volatility (6-month rolling std)'); ax.set_ylabel('Volatility (%)'); ax.legend()
save_fig('fig12_commodity_volatility')

# ===== FIG 13: Debt distributions =====
fig,axes=plt.subplots(1,2,figsize=(14,5))
for i,yr in enumerate([2019,2023]):
    d=annual[(annual['year']==yr)&annual['debt_gdp'].notna()&annual['country_group'].notna()]
    for g,clr in GC.items():
        s=d[d['country_group']==g]; axes[i].hist(s['debt_gdp'].clip(0,200),bins=25,alpha=0.6,color=clr,label=g,edgecolor='white')
    med=d['debt_gdp'].median(); axes[i].axvline(x=med,color='black',ls='--',lw=1,label=f'Median: {med:.0f}%')
    axes[i].set_title(f'Figure 13{"ab"[i]}: Debt/GDP ({yr})'); axes[i].set_xlabel('Debt/GDP (%)'); axes[i].legend(fontsize=8); axes[i].set_xlim(0,200)
plt.tight_layout(); save_fig('fig13_debt_distributions')

# ===== TABLES =====
print("\n"+"="*60+"\nGENERATING TABLES\n"+"="*60)

# Table 1: Summary stats
ma=annual[(annual['year']>=2000)&annual['country_group'].notna()]
sr=[]
for g in ['Advanced economies','EMDEs','All']:
    s=ma if g=='All' else ma[ma['country_group']==g]
    for v,l in [('hcpi','Headline CPI'),('fcpi','Food CPI'),('ecpi','Energy CPI'),('gdp_deflator','GDP deflator'),
                ('debt_gdp','Govt debt/GDP'),('fiscal_balance','Fiscal balance'),('ext_debt_gdp','Ext.debt/GDP'),('private_credit_gdp','Credit/GDP')]:
        d=s[v].dropna()
        if len(d)>0: sr.append({'Group':g,'Variable':l,'N':len(d),'Mean':round(d.mean(),2),'Median':round(d.median(),2),
                                 'Std':round(d.std(),2),'Min':round(d.min(),2),'Max':round(d.max(),2)})
pd.DataFrame(sr).to_csv(os.path.join(TDIR,'table01_summary_statistics.csv'),index=False)
print("  Table 1: Summary statistics saved")

# Table 2: Country classifications
ci=monthly[monthly['country_group'].notna()][['country_code','country_group','region','income_group']].drop_duplicates()
pd.crosstab(ci['region'],ci['income_group'],margins=True).to_csv(os.path.join(TDIR,'table02_country_classification.csv'))
print("  Table 2: Country classifications saved")

# Table 3: Correlations
ca=annual[(annual['year']>=2000)&annual['hcpi'].notna()]
cr=[]
for v,l in [('debt_gdp','Debt/GDP'),('fiscal_balance','Fiscal bal.'),('ext_debt_gdp','Ext.debt/GDP'),
            ('private_credit_gdp','Credit/GDP'),('sovereign_rating','Sov.rating'),('oil_price','Oil price')]:
    if v in ca.columns:
        vd=ca[['hcpi',v]].dropna()
        if len(vd)>30: cr.append({'Variable':l,'Corr_with_inflation':round(vd['hcpi'].corr(vd[v]),3),'N':len(vd)})
pd.DataFrame(cr).sort_values('Corr_with_inflation',key=abs,ascending=False).to_csv(os.path.join(TDIR,'table03_correlations.csv'),index=False)
print("  Table 3: Correlations saved")

print(f"\n{'='*60}\nSTEP 4 COMPLETE\n{'='*60}")
print(f"\n  {fc} figures saved to: {os.path.abspath(FDIR)}/")
print(f"  3 tables saved to: {os.path.abspath(TDIR)}")
