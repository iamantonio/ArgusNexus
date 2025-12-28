/**
 * ARGUSNEXUS V4 | Terminal Application
 * Bloomberg-style trading dashboard controller
 */

class ArgusTerminal {
    constructor() {
        this.api = window.api;
        this.charts = window.chartManager;

        this.refreshInterval = 60000; // 60 seconds
        this.refreshTimer = null;
        this.strategy = 'coinbase_survivor';
        this.activePosition = null;

        this.init();
    }

    /**
     * Initialize the terminal
     */
    async init() {
        console.log('[TERMINAL] Initializing...');

        // Load theme preference
        this.loadTheme();

        // Bind event listeners
        this.bindEvents();

        // Initial data load
        await this.refresh();

        // Start auto-refresh
        this.startAutoRefresh();

        // Update timestamp every second
        this.startClock();

        console.log('[TERMINAL] Ready');
    }

    /**
     * Load saved theme preference
     */
    loadTheme() {
        const savedTheme = localStorage.getItem('argusnexus-theme');
        const isLight = savedTheme === 'light';
        if (isLight) {
            document.body.classList.add('light-mode');
            this.charts.setLightMode(true);
        }
        this.updateThemeIcon(isLight);
    }

    /**
     * Toggle between light and dark modes
     */
    toggleTheme() {
        const isLight = document.body.classList.toggle('light-mode');
        localStorage.setItem('argusnexus-theme', isLight ? 'light' : 'dark');
        this.updateThemeIcon(isLight);

        // Re-render chart with new colors
        this.charts.setLightMode(isLight);
        this.updateChart();
    }

    /**
     * Update theme toggle icon
     */
    updateThemeIcon(isLight) {
        const icon = document.getElementById('theme-icon');
        if (icon) {
            // Sun for dark mode (click to switch to light), Moon for light mode
            icon.innerHTML = isLight ? '&#9790;' : '&#9788;';
            icon.title = isLight ? 'Switch to Dark Mode' : 'Switch to Light Mode';
        }
    }

    /**
     * Bind UI event listeners
     */
    bindEvents() {
        // Theme toggle
        const themeToggle = document.getElementById('theme-toggle');
        if (themeToggle) {
            themeToggle.addEventListener('click', () => this.toggleTheme());
        }

        // Strategy selector
        const strategySelect = document.getElementById('strategy-select');
        if (strategySelect) {
            strategySelect.addEventListener('change', (e) => {
                this.strategy = e.target.value;
                this.refresh();
            });
        }

        // Chart symbol selector
        const symbolSelect = document.getElementById('chart-symbol');
        if (symbolSelect) {
            symbolSelect.addEventListener('change', (e) => {
                this.charts.setSymbol(e.target.value);
            });
        }

        // Chart timeframe selector
        const timeframeSelect = document.getElementById('chart-timeframe');
        if (timeframeSelect) {
            timeframeSelect.addEventListener('change', (e) => {
                this.charts.setTimeframe(e.target.value);
            });
        }
    }

    /**
     * Refresh all data
     */
    async refresh() {
        console.log('[TERMINAL] Refreshing data...');

        try {
            await Promise.all([
                this.updateScoreboard(),
                this.updateTicker(),
                this.updateFleet(),
                this.updateHistory(),
                this.updateDecisions(),
                this.updateChart()
            ]);

            this.updateLastRefresh();
            this.setConnectionStatus(true);

        } catch (error) {
            console.error('[TERMINAL] Refresh failed:', error);
            this.setConnectionStatus(false);
        }
    }

    /**
     * Update scoreboard metrics
     */
    async updateScoreboard() {
        try {
            const data = await this.api.getScoreboard(this.strategy);
            const balance = await this.api.getBalance();

            // Total Equity
            this.updateMetric('metric-equity', {
                value: this.formatCurrency(balance.total_equity || 0),
                valueClass: 'positive',
                sub: 'PORTFOLIO VALUE'
            });

            // Cash
            this.updateMetric('metric-cash', {
                value: this.formatCurrency(balance.cash || 0),
                sub: 'USD BALANCE'
            });

            // BTC Value
            this.updateMetric('metric-btc', {
                value: this.formatCurrency(balance.btc_value || 0),
                sub: `${(balance.btc_qty || 0).toFixed(6)} BTC`
            });

            // Total P&L
            this.updateMetric('metric-total', {
                value: this.formatCurrency(data.total_pnl),
                valueClass: data.total_pnl >= 0 ? 'positive' : 'negative',
                sub: 'NET PERFORMANCE'
            });

            // Win Rate
            this.updateMetric('metric-winrate', {
                value: `${data.win_rate.toFixed(1)}%`,
                sub: `${data.wins}W / ${data.losses}L`
            });

            // Fleet
            this.updateMetric('metric-fleet', {
                value: data.active_positions.toString(),
                sub: 'ACTIVE POSITIONS'
            });

            // Store positions for chart
            if (data.positions && data.positions.length > 0) {
                this.activePosition = data.positions[0];
            } else {
                this.activePosition = null;
            }

        } catch (error) {
            console.error('[SCOREBOARD] Update failed:', error);
        }
    }

    /**
     * Update a metric card
     */
    updateMetric(elementId, { value, valueClass, sub }) {
        const el = document.getElementById(elementId);
        if (!el) return;

        const valueEl = el.querySelector('.bb-metric-value');
        const subEl = el.querySelector('.bb-metric-sub');

        if (valueEl) {
            valueEl.textContent = value;
            valueEl.className = 'bb-metric-value';
            if (valueClass) {
                valueEl.classList.add(valueClass);
            }
        }

        if (subEl && sub) {
            subEl.textContent = sub;
        }
    }

    /**
     * Update ticker strip
     */
    async updateTicker() {
        try {
            const data = await this.api.getTicker();
            const strip = document.getElementById('ticker-strip');
            if (!strip || !data.tickers) return;

            strip.innerHTML = data.tickers.map(t => `
                <div class="bb-ticker-item">
                    <span class="bb-ticker-symbol">${t.symbol}</span>
                    <span class="bb-ticker-price">$${t.price.toLocaleString()}</span>
                    <span class="bb-ticker-change ${t.change_24h >= 0 ? 'positive' : 'negative'}">
                        ${t.change_24h >= 0 ? '+' : ''}${t.change_24h.toFixed(2)}%
                    </span>
                </div>
            `).join('');

        } catch (error) {
            console.error('[TICKER] Update failed:', error);
        }
    }

    /**
     * Update fleet status table
     */
    async updateFleet() {
        try {
            const data = await this.api.getScoreboard(this.strategy);
            const tbody = document.getElementById('fleet-body');
            const countBadge = document.getElementById('fleet-count');

            if (!tbody) return;

            const positions = data.positions || [];

            // Update count badge
            if (countBadge) {
                countBadge.textContent = positions.length.toString();
            }

            if (positions.length === 0) {
                tbody.innerHTML = `
                    <tr class="bb-table-empty">
                        <td colspan="6">NO ACTIVE POSITIONS</td>
                    </tr>
                `;
                this.updatePositionDetail(null);
                return;
            }

            tbody.innerHTML = positions.map(p => `
                <tr>
                    <td>${p.symbol}</td>
                    <td class="${p.side === 'buy' ? 'bb-buy' : 'bb-sell'}">${p.side.toUpperCase()}</td>
                    <td>$${p.entry_price.toLocaleString()}</td>
                    <td>$${p.current_price.toLocaleString()}</td>
                    <td class="${p.unrealized_pnl >= 0 ? 'bb-profit' : 'bb-loss'}">
                        ${p.unrealized_pnl >= 0 ? '+' : ''}$${p.unrealized_pnl.toFixed(2)}
                        <br><small>(${p.unrealized_pnl_pct >= 0 ? '+' : ''}${p.unrealized_pnl_pct.toFixed(2)}%)</small>
                    </td>
                    <td class="bb-loss">$${p.stop_loss ? p.stop_loss.toLocaleString() : '--'}</td>
                </tr>
            `).join('');

            // Update position detail with first position
            this.updatePositionDetail(positions[0]);

            // Update chart symbol selector to match active position
            const symbolSelect = document.getElementById('chart-symbol');
            if (symbolSelect && positions[0]) {
                symbolSelect.value = positions[0].symbol;
            }

        } catch (error) {
            console.error('[FLEET] Update failed:', error);
        }
    }

    /**
     * Update position detail panel
     */
    updatePositionDetail(position) {
        const entry = document.getElementById('detail-entry');
        const stop = document.getElementById('detail-stop');
        const target = document.getElementById('detail-target');
        const risk = document.getElementById('detail-risk');

        if (!position) {
            if (entry) entry.textContent = '--';
            if (stop) stop.textContent = '--';
            if (target) target.textContent = '--';
            if (risk) risk.textContent = '--';
            return;
        }

        if (entry) entry.textContent = `$${position.entry_price.toLocaleString()}`;
        if (stop) stop.textContent = position.stop_loss ? `$${position.stop_loss.toLocaleString()}` : '--';
        if (target) target.textContent = position.take_profit ? `$${position.take_profit.toLocaleString()}` : '--';

        // Calculate risk
        if (risk && position.stop_loss && position.entry_price) {
            const riskPct = Math.abs((position.entry_price - position.stop_loss) / position.entry_price * 100);
            risk.textContent = `${riskPct.toFixed(2)}%`;
        } else if (risk) {
            risk.textContent = '--';
        }
    }

    /**
     * Update trade history table
     */
    async updateHistory() {
        try {
            const data = await this.api.getClosedTrades({
                strategy: this.strategy,
                limit: 20
            });

            const tbody = document.getElementById('history-body');
            const countBadge = document.getElementById('history-count');

            if (!tbody) return;

            // Update count badge
            if (countBadge) {
                countBadge.textContent = data.length.toString();
            }

            if (data.length === 0) {
                tbody.innerHTML = `
                    <tr class="bb-table-empty">
                        <td colspan="7">NO CLOSED TRADES</td>
                    </tr>
                `;
                return;
            }

            tbody.innerHTML = data.map(t => {
                const date = t.exit_timestamp ?
                    new Date(t.exit_timestamp).toLocaleDateString('en-US', {
                        month: 'short',
                        day: 'numeric',
                        hour: '2-digit',
                        minute: '2-digit'
                    }) : '--';

                return `
                    <tr>
                        <td>${date}</td>
                        <td>${t.symbol}</td>
                        <td class="${t.side === 'buy' ? 'bb-buy' : 'bb-sell'}">${t.side.toUpperCase()}</td>
                        <td>$${t.entry_price.toLocaleString()}</td>
                        <td>$${t.exit_price.toLocaleString()}</td>
                        <td class="${t.net_pnl >= 0 ? 'bb-profit' : 'bb-loss'}">
                            ${t.net_pnl >= 0 ? '+' : ''}$${t.net_pnl.toFixed(2)}
                        </td>
                        <td>${t.exit_reason || '--'}</td>
                    </tr>
                `;
            }).join('');

        } catch (error) {
            console.error('[HISTORY] Update failed:', error);
        }
    }

    /**
     * Update decision feed
     */
    async updateDecisions() {
        try {
            const data = await this.api.getDecisions({
                strategy: this.strategy,
                limit: 15
            });

            const feed = document.getElementById('decision-feed');
            const countBadge = document.getElementById('decision-count');

            if (!feed) return;

            // Update count badge
            if (countBadge) {
                countBadge.textContent = data.length.toString();
            }

            if (data.length === 0) {
                feed.innerHTML = `
                    <div class="bb-feed-empty">AWAITING DECISIONS...</div>
                `;
                return;
            }

            feed.innerHTML = data.map(d => {
                const time = d.timestamp ?
                    new Date(d.timestamp).toLocaleTimeString('en-US', {
                        hour: '2-digit',
                        minute: '2-digit',
                        second: '2-digit'
                    }) : '--';

                // Determine result class
                let resultClass = 'hold';
                if (d.result && d.result.includes('signal_long')) {
                    resultClass = 'signal';
                } else if (d.result && (d.result.includes('signal_close') || d.result.includes('signal_short'))) {
                    resultClass = 'close';
                }

                const resultText = d.result ? d.result.toUpperCase().replace(/_/g, ' ') : 'N/A';
                const reason = d.result_reason || 'No reason provided';

                return `
                    <div class="bb-feed-item">
                        <div class="bb-feed-header">
                            <span class="bb-feed-time">${time}</span>
                            <span class="bb-feed-result ${resultClass}">${resultText}</span>
                        </div>
                        <div class="bb-feed-reason">${reason.substring(0, 100)}${reason.length > 100 ? '...' : ''}</div>
                    </div>
                `;
            }).join('');

        } catch (error) {
            console.error('[DECISIONS] Update failed:', error);
        }
    }

    /**
     * Update chart
     */
    async updateChart() {
        const symbolSelect = document.getElementById('chart-symbol');
        const timeframeSelect = document.getElementById('chart-timeframe');

        const symbol = symbolSelect ? symbolSelect.value : 'BTC-USD';
        const timeframe = timeframeSelect ? timeframeSelect.value : 'FOUR_HOUR';

        // Pass active position for overlay lines
        const position = this.activePosition && this.activePosition.symbol === symbol ?
            this.activePosition : null;

        await this.charts.render('price-chart', symbol, timeframe, position);
    }

    /**
     * Start auto-refresh timer
     */
    startAutoRefresh() {
        if (this.refreshTimer) {
            clearInterval(this.refreshTimer);
        }

        this.refreshTimer = setInterval(() => {
            this.refresh();
        }, this.refreshInterval);
    }

    /**
     * Start clock
     */
    startClock() {
        const update = () => {
            const el = document.getElementById('timestamp');
            if (el) {
                const now = new Date();
                el.textContent = now.toUTCString().split(' ')[4] + ' UTC';
            }
        };

        update();
        setInterval(update, 1000);
    }

    /**
     * Update last refresh timestamp
     */
    updateLastRefresh() {
        const el = document.getElementById('status-last-update');
        if (el) {
            const now = new Date();
            el.textContent = `LAST: ${now.toLocaleTimeString('en-US', {
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit'
            })}`;
        }
    }

    /**
     * Set connection status
     */
    setConnectionStatus(connected) {
        const el = document.getElementById('status-connection');
        if (!el) return;

        if (connected) {
            el.innerHTML = '<span class="bb-status-dot"></span> CONNECTED';
            el.className = 'bb-status-item bb-status-ok';
        } else {
            el.innerHTML = '<span class="bb-status-dot"></span> DISCONNECTED';
            el.style.color = 'var(--bb-red)';
        }
    }

    /**
     * Format currency
     */
    formatCurrency(value) {
        const abs = Math.abs(value);
        const sign = value >= 0 ? '' : '-';
        return `${sign}$${abs.toLocaleString(undefined, {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        })}`;
    }
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
    window.terminal = new ArgusTerminal();
});
