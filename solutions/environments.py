import numpy as np
import pandas as pd

class SingleStockEnvironment:
    def __init__(self, df, episode_len = 600, short_window = 15, long_window = 60,
                state_cols = ['hodl','close_change','short','long','signal'], commission=0.0005):
        '''
        Default:
            * take input `df` which has at least `timestamp` and `close`
            * 0.05% commission; risk free rate of 0%
            * benchmark with buy and HODL and momentum strategies only
        '''
        self.episode_len = 600
        self.short_window = short_window
        self.long_window = long_window
        self.master_df = self.process_df(df.copy())
        self.state_cols = state_cols
        self.actions = [-1,0,1]
        self.commission = commission
        self.reset()
        
    def reset(self):
        '''
        Fill in the blank
        Sample a data frame with episode length `self.episode_len` from `self.master_df`
        return current state as a numpy array
        ''' 
        self.idx = 0
        self.previous_action = 0
        rand_idx = np.random.randint(0,self.master_df.shape[0]-self.episode_len)
        self.df = self.master_df.copy().iloc[rand_idx:(rand_idx+self.episode_len),:].reset_index(drop=True)
        return self.get_state()
    
    def step(self, action_idx):
        '''
        Fill in the black
        Default: take either `long` (1), `do nothing` (0) or `short` (-1) as action
        Possible: include `do nothing` (0) as liquidating all existing positions, especially
        interesting when taking into account commmissions
        Input: action_idx (don't forget to record it to `self.df`)
        Output: next state, reward, done, info (can be any useful information)
        '''
        #record action
        self.df.iloc[self.idx,self.df.columns.get_loc('signal')] = self.actions[action_idx]
        
        #record commission, if any
        if self.previous_action==0 and self.actions[action_idx]!=0: #change from nothing to position
            self.df.iloc[self.idx,self.df.columns.get_loc('commission')] = self.commission
        elif self.previous_action!=self.actions[action_idx] and self.previous_action!=0: #change from position to another position
            self.df.iloc[self.idx,self.df.columns.get_loc('commission')] = self.commission*2
        self.previous_action = self.actions[action_idx]
        
        #reward
        reward = self.get_reward()
        
        #done
        done=False
        if self.idx == self.df.index.max()-1: done=True
        
        #info
        info = {'idx':self.idx, 
                'reward': reward,
                'model returns': ((self.df.close_returns[:self.idx]+self.df.commission[:self.idx]) * self.df.signal[:self.idx] + 1).prod(),
                'HODL returns': (1+self.df.close_returns[:self.idx]).prod(),
                'SHORTL returns': (1-self.df.close_returns[:self.idx]).prod(),
                'mom returns': (self.df.close_returns[:self.idx] * self.df.mom_signal[:self.idx] + 1).prod()}
        
        #increment idx
        self.idx+=1

        return self.get_state(),reward,done, info
        
    def get_state(self):
        '''
        Fill in the blank
        Default: [HODL portfolio (from first period), close price change from last period, 
        short indicator, long indicator]
        Possible: indicators, previous timesteps
        '''
        return np.array(self.df.loc[self.idx,self.state_cols])
    
    def get_reward(self):
        '''
        Fill in the blank
        Default: returns of close price 
        Possible: sharpe ratio, sortino ratio, change in portfolio etc.
        '''
        rewards = ((self.df.close_returns+self.df.commission)*self.df.signal+1).cumprod().diff()
#         rewards = self.df.close_returns * self.df.signal
        return np.round(np.nan_to_num(rewards[self.idx])*1000,6)
#         return 1 if rewards[self.idx] > 0 else -1
        
    def process_df(self,df):
        #minimal features
        df['close_change'] = df.close.pct_change()
        df.fillna(0, inplace=True)
        df['hodl'] = (df.close_change+1).cumprod()
        #momentum features
        df['short'] = df.hodl.rolling(window=self.short_window, min_periods=1, center=False).mean()
        # df['short'] = df.hodl.ewm(span=self.short_window, min_periods=1).mean()
        df['long'] = df.hodl.rolling(window=self.long_window, min_periods=1, center=False).mean()
        # df['long'] = df.hodl.ewm(span=self.long_window, min_periods=1).mean()
        df['mom_signal'] = 0
        df.iloc[self.short_window:,df.columns.get_loc('mom_signal')] = np.where(df['short'][self.short_window:] > df['long'][self.short_window:], 1, -1)
        #returns
        df['close_returns'] = df.close_change.shift(-1)
        df['signal'] = 0
        df['commission'] = 0
        return df