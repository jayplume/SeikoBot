import os
import requests
# import pandas as pd
import numpy as np
import time
from dotenv import load_dotenv
from itertools import combinations
import matplotlib.pyplot as plt
import sys
from collections import defaultdict

load_dotenv()

API_KEY = os.getenv("RECALL_API_KEY")
BASE_URL = "https://api.sandbox.competitions.recall.network"
PRICE_ENDPOINT = f"{BASE_URL}/api/price"
BALANCES_ENDPOINT = f"{BASE_URL}/api/agent/balances"
TRADE_ENDPOINT = f"{BASE_URL}/api/trade/execute"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Mainnet token addresses (ETH)
TOKENS = {
    "USDC": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
    "WETH": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
    "DAI":  "0x6B175474E89094C44Da98b954EedeAC495271d0F"
}

def get_logger(logfile="seiko_agent.log"):
    class Logger:
        def __init__(self, logfile):
            self.logfile = logfile
            self.file = open(logfile, "a")
        def log(self, msg):
            tag = "[seiko] "
            print(tag + str(msg))
            self.file.write(tag + str(msg) + "\n")
            self.file.flush()
        def close(self):
            self.file.close()
    return Logger(logfile)

logger = get_logger()

class RecallAPI:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    # def get_price(self, token_address, chain="evm", specific_chain="eth"):
    #     params = {
    #         "token": token_address,
    #         "chain": chain,
    #         "specificChain": specific_chain
    #     }
    #     resp = self.session.get(PRICE_ENDPOINT, params=params)
    #     if resp.ok:
    #         return resp.json().get("price", None)
    #     return None

    def get_price(self, token_address, chain="evm", specific_chain="eth"):
        params = {
            "token": token_address,
            "chain": chain,
            "specificChain": specific_chain
        }
        resp = self.session.get(PRICE_ENDPOINT, params=params)
        if resp.ok:
            return resp.json().get("price", None)
        print("Failed price fetch:", resp.status_code, resp.text)
        return None

    def get_balances(self):
        resp = self.session.get(BALANCES_ENDPOINT)
        if resp.ok:
            return resp.json().get("balances", [])
        return []

    def place_order(self, from_token, to_token, amount, reason, from_chain="evm", from_specific_chain="eth", to_chain="evm", to_specific_chain="eth"):
        data = {
            "fromToken": from_token,
            "toToken": to_token,
            "amount": str(amount),
            "reason": reason,
            "fromChain": from_chain,
            "fromSpecificChain": from_specific_chain,
            "toChain": to_chain,
            "toSpecificChain": to_specific_chain
        }
        resp = self.session.post(TRADE_ENDPOINT, json=data)
        if resp.ok:
            return resp.json()
        print("[seiko] Trade API error:", resp.status_code, resp.text)
        return None

class MicrostructureAnalyzer:
    def analyze(self, price_history):
        if len(price_history) < 2:
            return None
        spread = np.max(price_history[-10:]) - np.min(price_history[-10:]) if len(price_history) >= 10 else np.max(price_history) - np.min(price_history)
        mid = price_history[-1]
        return {"spread": spread, "mid": mid}

class AdaptiveScalper:
    def __init__(self, min_size=1, max_size=10):
        self.min_size = min_size
        self.max_size = max_size
        self.price_history = []

    def update(self, price):
        self.price_history.append(price)
        if len(self.price_history) > 100:
            self.price_history.pop(0)

    def get_trade_size(self):
        if len(self.price_history) < 2:
            return self.min_size
        volatility = np.std(self.price_history)
        size = self.min_size + (self.max_size - self.min_size) * min(volatility / 0.01, 1)
        return max(self.min_size, min(self.max_size, size))

class TradeTracker:
    def __init__(self):
        self.trades = []
        self.total_profit = 0.0
        self.start_time = time.time()
        self.trades_executed = 0
        self.pnl_history = []
        self.cycle_history = []
        self.pair_pnl = defaultdict(float)
        self.pair_trades = defaultdict(list)
        self.pair_wins = defaultdict(int)
        self.pair_losses = defaultdict(int)

    def add_trade(self, from_token, to_token, from_amount, to_amount, price, trade_type, success):
        trade = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "from_token": from_token,
            "to_token": to_token,
            "from_amount": from_amount,
            "to_amount": to_amount,
            "price": price,
            "type": trade_type,
            "success": success
        }
        self.trades.append(trade)
        pair = f"{from_token}->{to_token}"
        pnl = (to_amount - from_amount) if success and from_amount and to_amount else 0
        if success and from_amount and to_amount:
            self.total_profit += pnl
            self.pair_pnl[pair] += pnl
            self.pair_trades[pair].append(pnl)
            if pnl > 0:
                self.pair_wins[pair] += 1
            elif pnl < 0:
                self.pair_losses[pair] += 1
        if success:
            self.trades_executed += 1

    def record_pnl(self, cycle):
        self.pnl_history.append(self.total_profit)
        self.cycle_history.append(cycle)

    def plot_pair_pnl(self):
        if not self.pair_pnl:
            return
        plt.figure(figsize=(8, 4))
        pairs = list(self.pair_pnl.keys())
        pnls = [self.pair_pnl[pair] for pair in pairs]
        plt.bar(pairs, pnls, color='skyblue')
        plt.title('SeikoAgent Per-Pair PnL')
        plt.xlabel('Pair')
        plt.ylabel('Cumulative PnL')
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig('seiko_pair_pnl.png')
        plt.close()
        logger.log("[seiko] [Pair PnL Chart] Saved to seiko_pair_pnl.png")

    def plot_pair_trades(self):
        if not self.pair_trades:
            return
        plt.figure(figsize=(8, 4))
        pairs = list(self.pair_trades.keys())
        counts = [len(self.pair_trades[pair]) for pair in pairs]
        plt.bar(pairs, counts, color='orange')
        plt.title('SeikoAgent Trades per Pair')
        plt.xlabel('Pair')
        plt.ylabel('Number of Trades')
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig('seiko_pair_trades.png')
        plt.close()
        logger.log("[seiko] [Pair Trades Chart] Saved to seiko_pair_trades.png")

    def print_stats(self):
        runtime = (time.time() - self.start_time) / 60
        logger.log("="*40)
        logger.log("SeikoAgent PERFORMANCE STATS")
        logger.log(f"Runtime: {runtime:.2f} min")
        logger.log(f"Trades Executed: {self.trades_executed}")
        logger.log(f"Total PnL (to_amount - from_amount): {self.total_profit:.6f}")
        logger.log("--- Per-Pair PnL ---")
        for pair, pnl in self.pair_pnl.items():
            n = len(self.pair_trades[pair])
            wins = self.pair_wins[pair]
            losses = self.pair_losses[pair]
            avg_trade = sum(self.pair_trades[pair])/n if n else 0
            logger.log(f"{pair}: PnL={pnl:.6f} | Trades={n} | Win={wins} | Loss={losses} | Avg={avg_trade:.6f}")
        logger.log("="*40)
        self.plot_pnl()
        self.plot_pair_pnl()
        self.plot_pair_trades()

    def plot_pnl(self):
        if len(self.cycle_history) < 2:
            return
        plt.figure(figsize=(8, 4))
        plt.plot(self.cycle_history, self.pnl_history, marker='o')
        plt.title('SeikoAgent PnL Over Time')
        plt.xlabel('Cycle')
        plt.ylabel('Cumulative PnL')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('seiko_pnl.png')
        plt.close()
        logger.log("[PnL Chart] Saved to seiko_pnl.png")

class SeikoAgent:
    def __init__(self, tokens, chain="evm", specific_chain="eth", interval=10):
        self.api = RecallAPI()
        self.tokens = tokens
        self.chain = chain
        self.specific_chain = specific_chain
        self.analyzers = {pair: MicrostructureAnalyzer() for pair in combinations(tokens.keys(), 2)}
        self.scalpers = {pair: AdaptiveScalper() for pair in combinations(tokens.keys(), 2)}
        self.price_histories = {pair: [] for pair in combinations(tokens.keys(), 2)}
        self.interval = interval
        self.trade_tracker = TradeTracker()

    def print_balances(self):
        balances = self.api.get_balances()
        logger.log("Current Balances:")
        for entry in balances:
            symbol = None
            for k, v in self.tokens.items():
                if v.lower() == entry['tokenAddress'].lower():
                    symbol = k
            logger.log(f"  {symbol or entry['tokenAddress']}: {entry['amount']} on {entry['specificChain']}")

    def run(self, iterations=1000):
        pairs = list(combinations(self.tokens.keys(), 2))
        for i in range(iterations):
            logger.log(f"\nCycle {i+1}")
            self.print_balances()
            for pair in pairs:
                from_symbol, to_symbol = pair
                from_token = self.tokens[from_symbol]
                to_token = self.tokens[to_symbol]
                try:
                    price = self.api.get_price(from_token, self.chain, self.specific_chain)
                    if price is None:
                        logger.log(f"Price unavailable for {from_symbol}")
                        continue
                    self.scalpers[pair].update(price)
                    self.price_histories[pair].append(price)
                    if len(self.price_histories[pair]) > 100:
                        self.price_histories[pair].pop(0)
                    analysis = self.analyzers[pair].analyze(self.price_histories[pair])
                    if not analysis:
                        logger.log(f"Not enough data for analysis for {from_symbol}->{to_symbol}")
                        continue
                    trade_size = self.scalpers[pair].get_trade_size()
                    if analysis["spread"] < 0.02:
                        logger.log(f"[{from_symbol}->{to_symbol}] Placing buy and sell for {trade_size} units at price {price}")
                        buy_result = self.api.place_order(
                            from_token, to_token, trade_size, f"Seiko scalping buy {from_symbol}->{to_symbol}",
                            from_chain=self.chain, from_specific_chain=self.specific_chain,
                            to_chain=self.chain, to_specific_chain=self.specific_chain
                        )
                        logger.log(f"Buy result: {buy_result}")
                        if buy_result and buy_result.get('success'):
                            tx = buy_result.get('transaction', {})
                            self.trade_tracker.add_trade(
                                from_token, to_token,
                                tx.get('fromAmount', trade_size),
                                tx.get('toAmount', 0),
                                tx.get('price', price),
                                'buy', True
                            )
                        else:
                            self.trade_tracker.add_trade(from_token, to_token, trade_size, 0, price, 'buy', False)
                        sell_result = self.api.place_order(
                            to_token, from_token, trade_size, f"Seiko scalping sell {to_symbol}->{from_symbol}",
                            from_chain=self.chain, from_specific_chain=self.specific_chain,
                            to_chain=self.chain, to_specific_chain=self.specific_chain
                        )
                        logger.log(f"Sell result: {sell_result}")
                        if sell_result and sell_result.get('success'):
                            tx = sell_result.get('transaction', {})
                            self.trade_tracker.add_trade(
                                to_token, from_token,
                                tx.get('fromAmount', trade_size),
                                tx.get('toAmount', 0),
                                tx.get('price', price),
                                'sell', True
                            )
                        else:
                            self.trade_tracker.add_trade(to_token, from_token, trade_size, 0, price, 'sell', False)
                    else:
                        logger.log(f"[{from_symbol}->{to_symbol}] Spread too wide: {analysis['spread']}")
                except Exception as e:
                    logger.log(f"Error for pair {from_symbol}->{to_symbol}: {e}")
            self.trade_tracker.record_pnl(i+1)
            if (i+1) % 10 == 0:
                self.trade_tracker.print_stats()
            time.sleep(self.interval)

def print_banner():
    logger.log("="*50)
    logger.log("  _____     _     _        ")
    logger.log(" / ____|   | |   | |       ")
    logger.log("| (___   __| | __| | ___   _ _ __ ")
    logger.log(" \\___ \\ / _` |/ _` |/ / | | | '_ \\ ")
    logger.log(" ____) | (_| | (_|   <| |_| | | | |")
    logger.log("|_____/ \\__,_|\\__,_|\\_\\__,_|_| |_|")
    logger.log("Seiko: The Precision Scalper")
    logger.log("="*50)

def print_bot_state(tokens, chain, specific_chain, interval, iterations):
    logger.log("Bot Configuration:")
    logger.log(f"  Tokens: {', '.join(tokens.keys())}")
    logger.log(f"  Chain: {chain}")
    logger.log(f"  Specific Chain: {specific_chain}")
    logger.log(f"  Interval: {interval} seconds")
    logger.log(f"  Iterations: {iterations}")
    logger.log("="*50)

def check_health():
    import requests
    url = "https://api.competitions.recall.network/api/health"
    try:
        response = requests.get(url, timeout=10)
        if response.ok:
            logger.log("\033[92m[Recall API Health: OK]\033[0m " + response.text)
        else:
            logger.log("\033[91m[Recall API Health: ERROR]\033[0m " + response.text)
    except Exception as e:
        logger.log(f"\033[91m[Recall API Health: EXCEPTION]\033[0m {e}")
    logger.log("="*50)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Seiko: The Precision Scalper")
    parser.add_argument("--chain", type=str, default="evm", help="Chain type (default: evm)")
    parser.add_argument("--specific_chain", type=str, default="eth", help="Specific chain (default: eth)")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of trading iterations")
    parser.add_argument("--interval", type=int, default=10, help="Seconds between each trading cycle (default: 10)")
    args = parser.parse_args()

    print_banner()
    print_bot_state(TOKENS, args.chain, args.specific_chain, args.interval, args.iterations)
    check_health()

    agent = SeikoAgent(
        tokens=TOKENS,
        chain=args.chain,
        specific_chain=args.specific_chain,
        interval=args.interval
    )
    agent.run(iterations=args.iterations) 
