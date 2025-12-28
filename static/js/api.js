/**
 * ARGUSNEXUS V4 | API Client
 * Fetch wrapper for all backend endpoints
 */

class ArgusAPI {
    constructor(baseUrl = '/api') {
        this.baseUrl = baseUrl;
    }

    /**
     * Generic GET request
     */
    async get(endpoint, params = {}) {
        const url = new URL(`${this.baseUrl}${endpoint}`, window.location.origin);

        Object.entries(params).forEach(([key, value]) => {
            if (value !== undefined && value !== null) {
                url.searchParams.set(key, value);
            }
        });

        try {
            const response = await fetch(url);
            if (!response.ok) {
                throw new Error(`API Error: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            console.error(`[API] ${endpoint} failed:`, error);
            throw error;
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // SCOREBOARD
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Get aggregated scoreboard metrics
     */
    async getScoreboard(strategy = null) {
        return this.get('/scoreboard', { strategy });
    }

    /**
     * Get daily performance breakdown
     */
    async getDailyPerformance(days = 30, strategy = null) {
        return this.get('/performance/daily', { days, strategy });
    }

    /**
     * Get portfolio balance
     */
    async getBalance() {
        return this.get('/balance');
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // TRADES
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Get open positions
     */
    async getOpenTrades(strategy = null) {
        return this.get('/trades/open', { strategy });
    }

    /**
     * Get closed trade history
     */
    async getClosedTrades(options = {}) {
        const { strategy, limit = 50, offset = 0 } = options;
        return this.get('/trades/closed', { strategy, limit, offset });
    }

    /**
     * Get trade details
     */
    async getTradeDetail(tradeId) {
        return this.get(`/trades/${tradeId}`);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // DECISIONS
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Get recent strategy decisions
     */
    async getDecisions(options = {}) {
        const { strategy, symbol, result, limit = 20, offset = 0 } = options;
        return this.get('/decisions', { strategy, symbol, result, limit, offset });
    }

    /**
     * Get decision statistics
     */
    async getDecisionStats(strategy = null, days = 7) {
        return this.get('/decisions/stats', { strategy, days });
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // PRICES
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Get current price for a symbol
     */
    async getPrice(symbol) {
        return this.get(`/prices/${symbol}`);
    }

    /**
     * Get OHLCV candle data
     */
    async getCandles(symbol, options = {}) {
        const { granularity = 'FOUR_HOUR', limit = 150 } = options;
        return this.get(`/candles/${symbol}`, { granularity, limit });
    }

    /**
     * Get ticker data for multiple symbols
     */
    async getTicker() {
        return this.get('/ticker');
    }
}

// Export singleton instance
window.api = new ArgusAPI();
