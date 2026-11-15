"""Synthetic, proposal-aligned staffing analysis. Not a clinical staffing system."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import milp, Bounds, LinearConstraint
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED=42
RATES={'regular':60.,'overtime':90.,'float':110.}
BUDGET=125000
OBJECTIVES=['staffing_cost','coverage_pct']

def pareto_mask(values, maximize=(False, True)):
    """Exact 2-objective skyline, O(n log n), with duplicates and ties retained.

    A dominates B iff A is no worse on each objective and strictly better on
    at least one. Directions are explicit; NaN/inf rejected. No tolerance or
    rounding changes the definition. All identical efficient rows remain.
    """
    v=np.asarray(values,dtype=float)
    if v.ndim!=2 or v.shape[1]!=2 or len(maximize)!=2:
        raise ValueError('Expected an n-by-2 array and two directions')
    if not np.isfinite(v).all():raise ValueError('Nonfinite objectives')
    out=np.zeros(len(v),bool)
    if not len(v):return out
    u=v*np.where(maximize,1.,-1.)
    order=np.lexsort((-u[:,1],-u[:,0]));best=-np.inf;i=0
    while i<len(order):
        j=i+1
        while j<len(order) and u[order[j],0]==u[order[i],0]:j+=1
        group=order[i:j];top=u[group,1].max()
        if top>best:out[group[u[group,1]==top]]=True
        best=max(best,top);i=j
    return out

def brute_pareto(values, maximize=(False,True)):
    v=np.asarray(values)*np.where(maximize,1,-1)
    return np.array([not any(np.all(a>=b) and np.any(a>b) for a in v) for b in v])

def generate_inputs():
    """Keep seeds separate so inputs do not change when candidate count changes."""
    rng=np.random.default_rng(SEED)
    shifts=pd.DataFrame({'shift_id':np.arange(28),'day':np.repeat(np.arange(14),2),'kind':np.tile(['Day','Night'],14)})
    shifts['forecast_census']=np.clip(np.rint(rng.normal(18,1.2,28)),12,24).astype(int)
    shifts['forecast_acuity']=rng.binomial(1,.18,28)
    shifts['required_rns']=np.ceil(shifts.forecast_census/5).astype(int)+shifts.forecast_acuity+1
    nurses=pd.DataFrame({'nurse_id':np.arange(22),'kind':np.repeat(['Day','Night'],11),'licensed':True,'competent':True})
    available=(rng.random((22,28))>.04)&(nurses.kind.to_numpy()[:,None]==shifts.kind.to_numpy()[None,:])
    sim=np.random.default_rng(SEED+1)
    # Day-level shared census shocks introduce correlation between shifts.
    census=np.clip(np.rint(18+sim.normal(0,2,(100,14)).repeat(2,axis=1)+sim.normal(0,1.5,(100,28))),12,24).astype(int)
    acuity=sim.binomial(1,.35,(100,28));need=np.ceil(census/5).astype(int)+acuity+1
    absent=sim.random((100,22,28))<.05
    scenarios=pd.DataFrame({'scenario_id':np.repeat(np.arange(100),28),'shift_id':np.tile(np.arange(28),100),'census':census.ravel(),'extra_acuity_rn':acuity.ravel(),'required_rns':need.ravel()})
    return shifts,nurses,available,absent,need,scenarios

def build_roster(counts,available,rng):
    """Binary assignments with availability, 48h/week caps, fixed shift cohorts.

    Dedicated Day/Night cohorts imply 24h between starts and at least 12h
    between the prior 12h shift end and next shift start. OT starts above 3
    shifts/week, counted per nurse. Overtime variables have premium cost.
    """
    n=22*28;nv=n+44;A=[];lo=[];hi=[]
    for s in range(28):
        row=np.zeros(nv);row[np.arange(22)*28+s]=1;A.append(row);lo.append(counts[s]);hi.append(counts[s])
    for nurse in range(22):
        for week in range(2):
            row=np.zeros(nv);row[nurse*28+np.arange(week*14,(week+1)*14)]=1
            A.append(row.copy());lo.append(0);hi.append(4)
            row[n+nurse*2+week]=-1;A.append(row);lo.append(-np.inf);hi.append(3)
    costs=np.r_[rng.uniform(0,1,n),np.full(44,360.)]
    result=milp(costs,integrality=np.ones(nv),bounds=Bounds(np.zeros(nv),np.r_[available.ravel().astype(float),np.ones(44)]),constraints=LinearConstraint(np.array(A),lo,hi),options={'time_limit':10})
    if not result.success:raise RuntimeError('No roster found; mandatory constraints not relaxed: '+result.message)
    return np.rint(result.x[:n]).astype(int).reshape(22,28)

def validate_roster(roster,shifts,available):
    assert roster.shape==(22,28) and np.isin(roster,[0,1]).all()
    assert np.all(roster<=available)
    assert np.all(roster.sum(axis=0)>=shifts.required_rns.to_numpy())
    assert np.all(roster[:,:14].sum(axis=1)<=4) and np.all(roster[:,14:].sum(axis=1)<=4)
    # Slots begin every 12h; adjacent slots disallowed for modeled 12h rest.
    assert not np.any((roster[:,:-1]+roster[:,1:])>1)

def cost_parts(roster,floats,rates=RATES):
    weeks=np.c_[roster[:,:14].sum(axis=1),roster[:,14:].sum(axis=1)]
    regular_hours=int(np.minimum(weeks,3).sum()*12);ot_hours=int(np.maximum(weeks-3,0).sum()*12)
    return regular_hours,ot_hours,int(floats.sum()*12),regular_hours*rates['regular']+ot_hours*rates['overtime']+floats.sum()*12*rates['float']

def generate_candidates(shifts,available,absent,need):
    rng=np.random.default_rng(SEED+2);records=[];rosters=[];reserves=[]
    risk=need.mean(axis=0)-shifts.required_rns.to_numpy()
    for j in range(20):
        counts=shifts.required_rns.to_numpy().copy()
        # Seven staffing levels, with several schedules per level; no rows are
        # manufactured by perturbing objective values after evaluation.
        priority=np.argsort(-(risk+rng.normal(0,.4,28)))
        for s in priority[:j%7]:counts[s]+=1
        roster=build_roster(counts,available,rng);validate_roster(roster,shifts,available)
        for reserve_total in [0,4,8,14,22]:
            floats=np.zeros(28,int)
            # Greedy preallocation uses expected need, not realized absences.
            # diminishing score permits a maximum of two qualified reserves.
            for _ in range(reserve_total):
                scores=risk+.05*counts-floats*.65+rng.normal(0,.05,28);scores[floats==2]=-np.inf
                floats[np.argmax(scores)]+=1
            supply=(roster[None,:,:]*(~absent)).sum(axis=1)+floats
            success=supply>=need;coverage=100*success.mean();rh,oh,fh,cost=cost_parts(roster,floats)
            cid=f'P{len(records)+1:03d}'
            # Replicate-level means handle within-scenario shift correlation.
            reps=100*success.mean(axis=1);se=reps.std(ddof=1)/np.sqrt(100)
            records.append({'candidate_id':cid,'roster_variant':j,'extra_regular_shifts':j%7,'regular_rn_shifts':int(roster.sum()),'float_rn_shifts':reserve_total,'mean_float_per_shift':reserve_total/28,'regular_hours':rh,'overtime_hours':oh,'float_hours':fh,'staffing_cost':cost,'coverage_pct':coverage,'coverage_ci_low':max(0,coverage-1.984*se),'coverage_ci_high':min(100,coverage+1.984*se),'worst_scenario_coverage_pct':reps.min(),'budget_met':cost<=BUDGET,'baseline_coverage_met':True})
            rosters.append(roster);reserves.append(floats)
    return pd.DataFrame(records),np.array(rosters),np.array(reserves)

def validate_data(df,shifts,nurses,available,absent,need,rosters,reserves,scenarios):
    if len(df)!=100 or df.isna().any().any():raise ValueError('Candidate count or missing values')
    numeric=df.select_dtypes(include='number')
    if not np.isfinite(numeric).all().all():raise ValueError('Nonfinite data')
    assert df.candidate_id.is_unique and not df.duplicated().any()
    assert len(np.unique(np.c_[rosters.reshape(100,-1),reserves],axis=0))==100
    assert df.coverage_pct.between(0,100).all() and (df.staffing_cost>0).all()
    assert scenarios.census.between(12,24).all() and len(scenarios)==2800
    assert np.isin(reserves,[0,1,2]).all() and len(nurses)==22
    assert nurses.licensed.all() and nurses.competent.all()
    for i in range(len(df)):
        validate_roster(rosters[i],shifts,available)
        assert abs(cost_parts(rosters[i],reserves[i])[3]-df.iloc[i].staffing_cost)<1e-6
        observed=100*(((rosters[i][None,:,:]*(~absent)).sum(axis=1)+reserves[i])>=need).mean()
        assert abs(observed-df.iloc[i].coverage_pct)<1e-8
    return {'rows':len(df),'missing_cells':int(df.isna().sum().sum()),'dtypes':df.dtypes.astype(str).to_dict(),'candidate_duplicates':int(df.duplicated().sum()),'distinct_full_assignment_decisions':100,'observed_census_mean':float(scenarios.census.mean()),'absence_rate':float(absent.mean()),'mandatory_roster_checks':'100/100 pass','cost_recalculation':'100/100 pass','coverage_recalculation':'100/100 pass','budget_met_count':int(df.budget_met.sum()),'scope':'Synthetic assumptions, not Kaiser observations'}

def analyze(df):
    df=df.copy();df['pareto']=pareto_mask(df[OBJECTIVES].values)
    assert np.array_equal(df.pareto,brute_pareto(df[OBJECTIVES].values))
    # Both stakeholder utilities use fixed observed-data anchors for this run.
    df['cost_utility']=100*(df.staffing_cost.max()-df.staffing_cost)/(df.staffing_cost.max()-df.staffing_cost.min())
    df['coverage_utility']=100*(df.coverage_pct-df.coverage_pct.min())/(df.coverage_pct.max()-df.coverage_pct.min())
    df['balanced_score']=df.cost_utility+df.coverage_utility
    df['equity_gap']=(df.cost_utility-df.coverage_utility).abs()
    df['weighted_score']=.3*df.cost_utility+.7*df.coverage_utility
    f=df[df.pareto]
    specs=[('Balanced','balanced_score',False),('Equity','equity_gap',True),('Cost priority','staffing_cost',True),('Coverage priority','coverage_pct',False),('Coverage weighted 70%','weighted_score',False)]
    selected=[]
    for strategy,key,ascending in specs:
        best=f.sort_values([key,'balanced_score','candidate_id'],ascending=[ascending,False,True]).iloc[0].copy();best['strategy']=strategy;selected.append(best)
    strategies=pd.DataFrame(selected)
    examples=[]
    for _,bad in df[~df.pareto].sort_values('balanced_score',ascending=False).iterrows():
        better=f[(f.staffing_cost<bad.staffing_cost)&(f.coverage_pct>bad.coverage_pct)]
        if len(better):
            good=better.sort_values('balanced_score',ascending=False).iloc[0]
            examples.append({'dominated':bad.candidate_id,'efficient_alternative':good.candidate_id,'cost_saving':bad.staffing_cost-good.staffing_cost,'coverage_gain_pp':good.coverage_pct-bad.coverage_pct})
        if len(examples)==5:break
    summary=df.groupby('pareto')[['staffing_cost','coverage_pct','float_rn_shifts','overtime_hours']].agg(['count','min','median','mean','max'])
    # Explicit scalar ranking; no natural best/worst ordering exists in 2D.
    extremes=pd.concat([df[~df.pareto].nlargest(1,'balanced_score').assign(comparison='Best dominated by balanced score'),f.nsmallest(1,'balanced_score').assign(comparison='Lowest-scoring efficient by balanced score')])
    return df,strategies,pd.DataFrame(examples),summary,extremes

def plot(df,strategies,root):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#0072B2','#D55E00','#009E73','#CC79A7','#6A3D9A'];marks=['o','s','^','D','P']
    fig,ax=plt.subplots(1,3,figsize=(19,6),layout='constrained')
    for flag,label,color,marker in [(False,'Dominated','#A6ABB3','x'),(True,'Pareto efficient','#0072B2','o')]:
        g=df[df.pareto==flag];ax[0].scatter(g.regular_rn_shifts,g.float_rn_shifts,c=color,marker=marker,label=label,alpha=.75,s=44)
        ax[1].scatter(g.staffing_cost/1000,g.coverage_pct,c=color,marker=marker,label=label,alpha=.65,s=38)
    f=df[df.pareto].sort_values('staffing_cost');ax[1].plot(f.staffing_cost/1000,f.coverage_pct,color='#0072B2',lw=1,alpha=.4,label='Discrete frontier guide')
    ax[1].axvline(BUDGET/1000,color='#555555',ls=':',lw=1,label='Assumed $125k ceiling')
    for i,(_,r) in enumerate(strategies.iterrows()):
        ax[1].scatter(r.staffing_cost/1000,r.coverage_pct,s=110,marker=marks[i],color=colors[i],edgecolor='black',label=f'{i+1} {r.strategy}',zorder=5)
        offset=(-20,14) if i==0 else ((7,14) if i==4 else (5,7))
        ax[1].annotate(str(i+1),(r.staffing_cost/1000,r.coverage_pct),xytext=offset,textcoords='offset points')
    ax[0].set(xlabel='Regular RN assignments across 28 shifts',ylabel='Reserved float RN shifts',title='1  Staffing decision space')
    ax[1].set(xlabel='14-day planned staffing cost ($ thousands)',ylabel='Shift coverage without unplanned call-in (%)',title='2  Cost and reliability tradeoff')
    # Conditional means over an explicitly sampled design; not a causal curve.
    g=df.groupby('float_rn_shifts')[['staffing_cost','coverage_pct']].mean()
    ax[2].plot(g.index,g.coverage_pct,'o-',color='#0072B2',label='Mean coverage (%)');right=ax[2].twinx();right.plot(g.index,g.staffing_cost/1000,'s--',color='#D55E00',label='Mean cost ($k)')
    ax[2].set(xlabel='Reserved float RN shifts over 14 days',ylabel='Mean simulated coverage (%)',title='3  Primary lever and both objectives');right.set_ylabel('Mean staffing cost ($ thousands)',color='#D55E00')
    h,l=ax[2].get_legend_handles_labels();h2,l2=right.get_legend_handles_labels();ax[2].legend(h+h2,l+l2,loc='lower right')
    for a in ax[:2]:a.legend(loc='lower right' if a is ax[0] else 'lower right',fontsize=8);a.grid(alpha=.12)
    fig.suptitle('Kaiser Permanente case study — hypothetical medical-surgical unit\n100 synthetic candidate plans • common 100-scenario evaluation • lower cost / higher coverage',fontsize=14)
    fig.savefig(root/'images/three_panel_analysis.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    df.staffing_cost.hist(ax=axes[0,0],bins=15,color='#0072B2');axes[0,0].set(title='Cost distribution',xlabel='Dollars',ylabel='Candidates')
    df.coverage_pct.hist(ax=axes[0,1],bins=15,color='#009E73');axes[0,1].set(title='Reliability distribution',xlabel='Coverage %',ylabel='Candidates')
    axes[1,0].scatter(df.float_rn_shifts,df.coverage_pct,c=df.regular_rn_shifts,cmap='viridis');axes[1,0].set(title='Reserves and reliability',xlabel='Float RN shifts',ylabel='Coverage %')
    cols=['regular_rn_shifts','float_rn_shifts','overtime_hours','staffing_cost','coverage_pct'];corr=df[cols].corr();im=axes[1,1].imshow(corr,vmin=-1,vmax=1,cmap='RdBu_r');axes[1,1].set_xticks(range(5),['Regular','Float','OT','Cost','Coverage'],rotation=35);axes[1,1].set_yticks(range(5),['Regular','Float','OT','Cost','Coverage']);axes[1,1].set_title('Candidate design correlations');fig.colorbar(im,ax=axes[1,1],shrink=.8)
    fig.savefig(root/'images/exploration.png',dpi=150);plt.close(fig)

def run(root):
    root=Path(root)
    for folder in ['data','images','results']: (root/folder).mkdir(exist_ok=True)
    shifts,nurses,available,absent,need,scenarios=generate_inputs()
    df,rosters,reserves=generate_candidates(shifts,available,absent,need)
    report=validate_data(df,shifts,nurses,available,absent,need,rosters,reserves,scenarios)
    df,strategies,examples,summary,extremes=analyze(df)
    shifts.to_csv(root/'data/shifts.csv',index=False);nurses.to_csv(root/'data/nurses.csv',index=False);scenarios.to_csv(root/'data/scenarios.csv',index=False)
    df.to_csv(root/'data/candidate_plans.csv',index=False)
    assignment=[]
    for i,r in df.iterrows():
        for s in range(28):assignment.append({'candidate_id':r.candidate_id,'shift_id':s,'nurse_ids':';'.join(map(str,np.flatnonzero(rosters[i,:,s]))),'float_rns':int(reserves[i,s])})
    pd.DataFrame(assignment).to_csv(root/'data/assignments.csv',index=False)
    np.savez_compressed(root/'data/simulation_inputs.npz',available=available,absent=absent,need=need,rosters=rosters,reserves=reserves)
    strategies.to_csv(root/'results/strategies.csv',index=False);examples.to_csv(root/'results/dominated_examples.csv',index=False);summary.to_csv(root/'results/group_summary.csv');extremes.to_csv(root/'results/scalar_comparison.csv',index=False)
    df.describe().to_csv(root/'results/descriptive_statistics.csv')
    sensitivity=[]
    for rate_name in ['regular','overtime','float']:
        for factor in [.8,1.,1.2]:
            rates=RATES.copy();rates[rate_name]*=factor
            costs=df.regular_hours*rates['regular']+df.overtime_hours*rates['overtime']+df.float_hours*rates['float']
            sensitivity.append({'rate_varied':rate_name,'rate_factor':factor,'efficient_count':int(pareto_mask(np.c_[costs,df.coverage_pct]).sum()),'budget_met_count':int((costs<=BUDGET).sum())})
    pd.DataFrame(sensitivity).to_csv(root/'results/rate_sensitivity.csv',index=False)
    report.update({'pareto_count':int(df.pareto.sum()),'dominated_count':int((~df.pareto).sum()),'pareto_matches_brute_force':True,'strict_improvement_examples':len(examples)})
    (root/'results/validation.json').write_text(json.dumps(report,indent=2));plot(df,strategies,root)
    return df,strategies,examples,summary,extremes,report

if __name__=='__main__':
    outputs=run(Path(__file__).resolve().parents[1]);print(json.dumps(outputs[-1],indent=2));print(outputs[1][['strategy','candidate_id','staffing_cost','coverage_pct','budget_met']].to_string(index=False))
