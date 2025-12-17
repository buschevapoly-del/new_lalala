
// data-loader.js (оптимизированная версия)
class DataLoader {
    constructor() {
        this.data = null;
        this.normalizedData = null;
        this.X_train = null;
        this.y_train = null;
        this.X_test = null;
        this.y_test = null;
        this.min = null;
        this.max = null;
        this.dateLabels = [];
        this.returns = [];
        this.trainIndices = [];
        this.testIndices = [];
        this.dataUrl = 'https://raw.githubusercontent.com/buschevapoly-del/again/main/my_data.csv';
        this.insights = {};
    }

    async loadCSVFromGitHub() {
        try {
            const response = await fetch(this.dataUrl);
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const content = await response.text();
            this.parseCSV(content);
            return this.data;
        } catch (error) {
            throw new Error(`Failed to load data: ${error.message}`);
        }
    }

    parseCSV(content) {
        const lines = content.trim().split('\n');
        const parsedData = [];
        this.dateLabels = [];
        this.returns = [];

        // Fast parsing with pre-allocated arrays
        for (let i = 1; i < lines.length; i++) {
            const line = lines[i].trim();
            if (!line) continue;

            const parts = line.split(';');
            if (parts.length >= 2) {
                const dateStr = parts[0].trim();
                const price = parseFloat(parts[1].trim());
                
                if (!isNaN(price) && price > 0) {
                    parsedData.push({ date: dateStr, price: price });
                    this.dateLabels.push(dateStr);
                }
            }
        }

        // Sort by date
        parsedData.sort((a, b) => {
            const dateA = this.parseDate(a.date);
            const dateB = this.parseDate(b.date);
            return dateA - dateB;
        });
        
        // Calculate returns efficiently
        const returns = new Array(parsedData.length - 1);
        for (let i = 1; i < parsedData.length; i++) {
            returns[i-1] = (parsedData[i].price - parsedData[i-1].price) / parsedData[i-1].price;
        }
        this.returns = returns;

        this.data = parsedData;
        
        // Calculate insights
        this.calculateInsights();
        
        if (this.data.length < 65) {
            throw new Error(`Insufficient data. Need at least 65 days, got ${this.data.length}`);
        }
    }

    calculateInsights() {
        if (!this.data || this.data.length === 0) return;
        
        const prices = this.data.map(d => d.price);
        const returns = this.returns;
        
        // 1. Basic Statistics
        const lastPrice = prices[prices.length - 1];
        const firstPrice = prices[0];
        const totalReturn = (lastPrice - firstPrice) / firstPrice;
        
        // 2. Daily Returns Statistics
        const meanReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
        const variance = returns.reduce((sq, n) => sq + Math.pow(n - meanReturn, 2), 0) / returns.length;
        const stdReturn = Math.sqrt(variance);
        const annualizedVolatility = stdReturn * Math.sqrt(252);
        
        // 3. Rolling Volatility (20-day)
        const window = 20;
        const rollingVolatilities = [];
        for (let i = window; i <= returns.length; i++) {
            const windowReturns = returns.slice(i - window, i);
            const windowMean = windowReturns.reduce((a, b) => a + b, 0) / window;
            const windowVar = windowReturns.reduce((sq, n) => sq + Math.pow(n - windowMean, 2), 0) / window;
            rollingVolatilities.push(Math.sqrt(windowVar) * Math.sqrt(252));
        }
        
        // 4. Trend Detection (Simple Moving Average Crossover)
        const sma50 = this.calculateSMA(prices, 50);
        const sma200 = this.calculateSMA(prices, 200);
        const currentTrend = sma50[sma50.length - 1] > sma200[sma200.length - 1] ? 'Bullish' : 'Bearish';
        
        // 5. Maximum Drawdown
        let maxDrawdown = 0;
        let peak = prices[0];
        for (let i = 1; i < prices.length; i++) {
            if (prices[i] > peak) peak = prices[i];
            const drawdown = (peak - prices[i]) / peak;
            if (drawdown > maxDrawdown) maxDrawdown = drawdown;
        }
        
        this.insights = {
            basic: {
                totalDays: this.data.length,
                dateRange: `${this.data[0].date} to ${this.data[this.data.length - 1].date}`,
                firstPrice: firstPrice.toFixed(2),
                lastPrice: lastPrice.toFixed(2),
                totalReturn: (totalReturn * 100).toFixed(2) + '%',
                maxDrawdown: (maxDrawdown * 100).toFixed(2) + '%'
            },
            returns: {
                meanDailyReturn: (meanReturn * 100).toFixed(4) + '%',
                stdDailyReturn: (stdReturn * 100).toFixed(4) + '%',
                annualizedVolatility: (annualizedVolatility * 100).toFixed(2) + '%',
                sharpeRatio: (meanReturn / stdReturn * Math.sqrt(252)).toFixed(2),
                positiveDays: ((returns.filter(r => r > 0).length / returns.length) * 100).toFixed(1) + '%'
            },
            trends: {
                currentTrend: currentTrend,
                sma50: sma50[sma50.length - 1].toFixed(2),
                sma200: sma200[sma200.length - 1].toFixed(2),
                aboveSMA200: (lastPrice > sma200[sma200.length - 1]) ? 'Yes' : 'No',
                trendStrength: Math.abs((sma50[sma50.length - 1] - sma200[sma200.length - 1]) / sma200[sma200.length - 1] * 100).toFixed(2) + '%'
            },
            volatility: {
                currentRollingVol: (rollingVolatilities[rollingVolatilities.length - 1] || 0).toFixed(2) + '%',
                avgRollingVol: (rollingVolatilities.reduce((a, b) => a + b, 0) / rollingVolatilities.length).toFixed(2) + '%',
                maxRollingVol: (Math.max(...rollingVolatilities) || 0).toFixed(2) + '%',
                minRollingVol: (Math.min(...rollingVolatilities) || 0).toFixed(2) + '%'
            },
            rollingVolatilities: rollingVolatilities,
            sma50: sma50,
            sma200: sma200
        };
    }
    
    calculateSMA(prices, period) {
        const sma = [];
        for (let i = period - 1; i < prices.length; i++) {
            const sum = prices.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0);
            sma.push(sum / period);
        }
        return sma;
    }

    parseDate(dateStr) {
        const parts = dateStr.split('.');
        if (parts.length === 3) {
            return new Date(parts[2], parts[1] - 1, parts[0]);
        }
        return new Date(dateStr);
    }

    prepareData(windowSize = 60, predictionHorizon = 5, testSplit = 0.2) {
        if (!this.returns || this.returns.length === 0) {
            throw new Error('No data available. Load CSV first.');
        }

        const totalSamples = this.returns.length - windowSize - predictionHorizon + 1;
        
        if (totalSamples <= 0) {
            throw new Error('Not enough data');
        }

        // Normalize returns
        this.normalizeReturns();

        // Create sequences using typed arrays for speed
        const sequences = new Array(totalSamples);
        const targets = new Array(totalSamples);

        for (let i = 0; i < totalSamples; i++) {
            sequences[i] = this.normalizedData.slice(i, i + windowSize).map(v => [v]);
            targets[i] = this.normalizedData.slice(i + windowSize, i + windowSize + predictionHorizon);
        }

        // Split chronologically
        const splitIdx = Math.floor(sequences.length * (1 - testSplit));
        this.trainIndices = Array.from({ length: splitIdx }, (_, i) => i);
        this.testIndices = Array.from({ length: sequences.length - splitIdx }, (_, i) => i + splitIdx);

        // Convert to tensors
        this.X_train = tf.tensor3d(sequences.slice(0, splitIdx), [splitIdx, windowSize, 1]);
        this.y_train = tf.tensor2d(targets.slice(0, splitIdx), [splitIdx, predictionHorizon]);
        this.X_test = tf.tensor3d(sequences.slice(splitIdx), [sequences.length - splitIdx, windowSize, 1]);
        this.y_test = tf.tensor2d(targets.slice(splitIdx), [sequences.length - splitIdx, predictionHorizon]);

        console.log(`Created ${sequences.length} samples: ${splitIdx} train, ${sequences.length - splitIdx} test`);
        
        return this;
    }

    normalizeReturns() {
        if (!this.returns || this.returns.length === 0) {
            throw new Error('No returns data available');
        }

        this.min = Math.min(...this.returns);
        this.max = Math.max(...this.returns);
        
        const range = this.max - this.min || 1;
        this.normalizedData = this.returns.map(ret => (ret - this.min) / range);
    }

    denormalize(value) {
        if (this.min === null || this.max === null) {
            throw new Error('Normalization parameters not available');
        }
        const range = this.max - this.min || 1;
        return value * range + this.min;
    }

    getHistoricalData() {
        if (!this.data) return null;
        
        return {
            dates: this.dateLabels,
            prices: this.data.map(d => d.price),
            returns: this.returns,
            normalizedReturns: this.normalizedData || []
        };
    }

    getDataSummary() {
        return this.insects.basic || null;
    }
    
    getInsights() {
        return this.insights;
    }

    dispose() {
        [this.X_train, this.y_train, this.X_test, this.y_test].forEach(tensor => {
            if (tensor) tensor.dispose();
        });
        this.X_train = this.y_train = this.X_test = this.y_test = this.normalizedData = null;
    }
}

export { DataLoader };
