// gru.js (обновленная версия)
class GRUModel {
    constructor(windowSize = 60, predictionHorizon = 5) {
        this.windowSize = windowSize;
        this.predictionHorizon = predictionHorizon;
        this.model = null;
        this.isTrained = false;
        this.batchSize = 32;
        this.trainingHistory = []; // Сохраняем историю обучения для расчета RMSE
    }

    buildModel() {
        if (this.model) {
            this.model.dispose();
        }
        
        tf.disposeVariables();
        
        this.model = tf.sequential();
        
        // Простая, но стабильная архитектура
        this.model.add(tf.layers.gru({
            units: 32,
            inputShape: [this.windowSize, 1],
            returnSequences: false,
            activation: 'tanh',
            kernelInitializer: 'glorotUniform',
            recurrentInitializer: 'orthogonal'
        }));
        
        // Dropout для регуляризации
        this.model.add(tf.layers.dropout({rate: 0.2}));
        
        // Промежуточный слой
        this.model.add(tf.layers.dense({
            units: 16,
            activation: 'relu',
            kernelInitializer: 'heNormal'
        }));
        
        // Выходной слой
        this.model.add(tf.layers.dense({
            units: this.predictionHorizon,
            activation: 'linear',
            kernelInitializer: 'glorotUniform'
        }));
        
        // Используем Adam с небольшим learning rate
        const optimizer = tf.train.adam(0.001);
        
        this.model.compile({
            optimizer: optimizer,
            loss: 'meanSquaredError',
            metrics: ['mse']
        });
        
        console.log('✅ GRU Model built');
        this.isTrained = false;
        this.trainingHistory = [];
        
        return this.model;
    }

    async train(X_train, y_train, epochs = 8, callbacks = {}) {
        console.log('Training GRU model...');
        
        if (!this.model) {
            this.buildModel();
        }
        
        if (!X_train || !y_train) {
            throw new Error('Training data not provided');
        }
        
        const sampleCount = X_train.shape[0];
        const batchSize = Math.min(this.batchSize, sampleCount);
        
        console.log(`Training: ${sampleCount} samples, batch=${batchSize}, epochs=${epochs}`);
        
        const startTime = Date.now();
        this.trainingHistory = []; // Сбрасываем историю
        
        try {
            // Простое обучение с сохранением истории
            const history = await this.model.fit(X_train, y_train, {
                epochs: epochs,
                batchSize: batchSize,
                validationSplit: 0.1,
                verbose: 0,
                shuffle: true
            });
            
            // Сохраняем историю для расчета RMSE
            this.trainingHistory = history.history;
            this.isTrained = true;
            
            console.log(`✅ Training completed in ${((Date.now() - startTime) / 1000).toFixed(1)}s`);
            console.log('Training history:', this.trainingHistory);
            
            // Рассчитываем RMSE из последнего значения MSE
            let rmse = 0;
            let mse = 0;
            
            if (this.trainingHistory.loss && this.trainingHistory.loss.length > 0) {
                mse = this.trainingHistory.loss[this.trainingHistory.loss.length - 1];
                rmse = Math.sqrt(Math.max(mse, 0));
            }
            
            if (callbacks.onTrainEnd) {
                callbacks.onTrainEnd(rmse);
            }
            
            return { 
                success: true, 
                rmse: rmse,
                mse: mse,
                history: this.trainingHistory 
            };
            
        } catch (error) {
            console.error('Training error:', error);
            this.isTrained = true; // Все равно разрешаем предсказания
            throw error;
        }
    }

    async predict(X) {
        if (!this.model) {
            this.buildModel();
        }
        
        try {
            const predictions = this.model.predict(X);
            const predictionsArray = await predictions.array();
            predictions.dispose();
            
            return predictionsArray;
        } catch (error) {
            console.error('Prediction error:', error);
            return [Array(this.predictionHorizon).fill(0)];
        }
    }

    // НОВЫЙ МЕТОД: расчет RMSE на тестовых данных
    async calculateRMSE(X_test, y_test) {
        if (!this.model || !this.isTrained) {
            return { rmse: 0, mse: 0, sampleCount: 0 };
        }

        try {
            // Делаем предсказания на тестовых данных
            const predictions = this.model.predict(X_test);
            const predArray = await predictions.array();
            const actualArray = await y_test.array();
            
            predictions.dispose();
            
            // Рассчитываем RMSE
            let totalSquaredError = 0;
            let sampleCount = 0;
            
            for (let i = 0; i < predArray.length; i++) {
                for (let j = 0; j < predArray[i].length; j++) {
                    const error = actualArray[i][j] - predArray[i][j];
                    totalSquaredError += error * error;
                    sampleCount++;
                }
            }
            
            const mse = totalSquaredError / Math.max(sampleCount, 1);
            const rmse = Math.sqrt(Math.max(mse, 0));
            
            return {
                rmse: rmse,
                mse: mse,
                sampleCount: sampleCount,
                predictions: predArray,
                actual: actualArray
            };
            
        } catch (error) {
            console.error('RMSE calculation error:', error);
            return { rmse: 0, mse: 0, sampleCount: 0 };
        }
    }

    // Метод для расчета RMSE на последних данных (более реалистичный)
    async calculateValidationRMSE(normalizedData, windowSize = 60, horizon = 5) {
        if (!this.model || !this.isTrained || !normalizedData) {
            return { rmse: 0, sampleCount: 0 };
        }
        
        try {
            // Используем последние 30% данных для валидации
            const validationSize = Math.floor(normalizedData.length * 0.3);
            const validationData = normalizedData.slice(-validationSize);
            
            const sequences = [];
            const targets = [];
            
            // Создаем последовательности для валидации
            for (let i = 0; i < validationData.length - windowSize - horizon; i++) {
                sequences.push(validationData.slice(i, i + windowSize).map(v => [v]));
                targets.push(validationData.slice(i + windowSize, i + windowSize + horizon));
            }
            
            if (sequences.length === 0) {
                return { rmse: 0, sampleCount: 0 };
            }
            
            // Создаем тензоры
            const X_val = tf.tensor3d(sequences, [sequences.length, windowSize, 1]);
            const y_val = tf.tensor2d(targets, [targets.length, horizon]);
            
            // Рассчитываем RMSE
            const result = await this.calculateRMSE(X_val, y_val);
            
            // Освобождаем память
            X_val.dispose();
            y_val.dispose();
            
            return result;
            
        } catch (error) {
            console.error('Validation RMSE error:', error);
            return { rmse: 0, sampleCount: 0 };
        }
    }

    getTrainingHistory() {
        return this.trainingHistory;
    }

    dispose() {
        if (this.model) {
            this.model.dispose();
            this.model = null;
        }
        this.isTrained = false;
        this.trainingHistory = [];
    }
}

export { GRUModel };
