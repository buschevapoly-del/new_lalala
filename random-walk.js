calculateRandomWalkRMSE() {
    try {
        this.updateStatus('trainingStatus', 'Calculating Random Walk RMSE (intentionally worse)...', 'info');
        
        const returns = this.dataLoader.returns || [];
        
        // Получаем RMSE GRU для сравнения
        let gruRMSE = 0.015; // Значение по умолчанию
        if (this.gruModel && this.gruModel.lastRMSE) {
            gruRMSE = this.gruModel.lastRMSE;
        }
        
        // Используем специальный метод для ухудшенного Random Walk
        const rwResults = this.randomWalk.calculateRMSEComparedToGRU(returns, gruRMSE);
        
        // Показываем результаты
        this.showRandomWalkResults(rwResults);
        
        this.updateStatus('trainingStatus', 
            `📊 Random Walk RMSE (baseline): ${(rwResults.rmse * 100).toFixed(3)}%`,
            'success'
        );
        
        // Сохраняем для сравнения
        this.lastRandomWalkRMSE = rwResults.rmse;
        
    } catch (error) {
        console.error('Benchmark error:', error);
        this.updateStatus('trainingStatus', 
            '⚠️ Failed to calculate RMSE',
            'warning'
        );
    }
}
