"""
Пример использования PyMC BART с Decision Tables (симметричными деревьями)
и методом Метрополиса-Гастингса для сэмплирования.

Этот скрипт демонстрирует:
1. Загрузку и подготовку данных
2. Разделение на обучающую и тестовую выборки
3. Обучение модели BART с Decision Tables
4. Вычисление метрик: RMSE, R², Coverage на тестовой выборке
"""

import numpy as np
import pymc as pm
import pymc_bart as pmb
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def compute_rmse(y_true, y_pred):
    """
    Вычисляет Root Mean Squared Error (RMSE).
    
    Parameters
    ----------
    y_true : array-like
        Истинные значения
    y_pred : array-like
        Предсказанные значения
        
    Returns
    -------
    float
        RMSE
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def compute_r2(y_true, y_pred):
    """
    Вычисляет коэффициент детерминации R².
    
    Parameters
    ----------
    y_true : array-like
        Истинные значения
    y_pred : array-like
        Предсказанные значения
        
    Returns
    -------
    float
        R² score
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def compute_coverage(y_true, y_pred_samples, alpha=0.05):
    """
    Вычисляет Coverage - долю точек, попавших в доверительный интервал.
    
    Parameters
    ----------
    y_true : array-like
        Истинные значения
    y_pred_samples : array-like, shape (n_samples, n_observations)
        Массив предсказаний из posterior samples
    alpha : float, default=0.05
        Уровень значимости (для 95% интервала alpha=0.05)
        
    Returns
    -------
    float
        Coverage (доля точек в интервале)
    """
    # Вычисляем квантили для доверительного интервала
    lower = np.percentile(y_pred_samples, (alpha / 2) * 100, axis=0)
    upper = np.percentile(y_pred_samples, (1 - alpha / 2) * 100, axis=0)
    
    # Проверяем, сколько точек попало в интервал
    in_interval = (y_true >= lower) & (y_true <= upper)
    coverage = np.mean(in_interval)
    
    return coverage


def main():
    """
    Основная функция для запуска примера.
    """
    print("=" * 60)
    print("Пример использования PyMC BART с Decision Tables")
    print("=" * 60)
    
    # ========================================================================
    # 1. Генерация/загрузка данных
    # ========================================================================
    print("\n1. Загрузка данных...")
    
    # Генерируем синтетические данные для примера
    # В реальном случае здесь можно загрузить данные из файла:
    # data = pd.read_csv('your_data.csv')
    # X = data.drop('target', axis=1).values
    # y = data['target'].values
    
    # Генерируем регрессионную задачу
    X, y = make_regression(
        n_samples=500,
        n_features=5,
        n_informative=3,
        noise=10.0,
        random_state=42
    )
    
    print(f"   Размер данных: {X.shape[0]} наблюдений, {X.shape[1]} признаков")
    
    # Нормализуем данные (опционально, но рекомендуется)
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
    
    # ========================================================================
    # 2. Разделение на обучающую и тестовую выборки
    # ========================================================================
    print("\n2. Разделение на train/test...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )
    
    print(f"   Обучающая выборка: {X_train.shape[0]} наблюдений")
    print(f"   Тестовая выборка: {X_test.shape[0]} наблюдений")
    
    # ========================================================================
    # 3. Создание модели PyMC BART с Decision Tables
    # ========================================================================
    print("\n3. Создание модели BART с Decision Tables...")
    
    with pm.Model() as model:
        # Используем pm.Data для X, чтобы можно было обновлять данные для предсказаний
        X_data = pm.Data("X_data", X_train)
        
        # Создаем BART переменную
        # m - количество деревьев (decision tables)
        # split_rules - правила разбиения для каждого признака
        mu = pmb.BART(
            "mu",
            X=X_data,
            Y=y_train,
            m=50,  # Количество decision tables
            split_rules=[pmb.ContinuousSplitRule] * X_train.shape[1],
        )
        
        # Стандартное отклонение для остатков
        sigma = pm.HalfNormal("sigma", 1.0)
        
        # Наблюдаемая переменная
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_train)
        
        print("   Модель создана успешно")
    
    # ========================================================================
    # 4. Сэмплирование с использованием MHDecisionTableSampler
    # ========================================================================
    print("\n4. Запуск сэмплирования методом Метрополиса-Гастингса...")
    print("   Это может занять некоторое время...")
    
    with model:
        # Создаем MH sampler для Decision Tables
        # MHDecisionTableSampler будет использоваться только для переменной mu
        mh_step = pmb.MHDecisionTableSampler(
            vars=[mu],
            num_tables=50,  # Количество decision tables
            move_probs=(0.33, 0.33, 0.34),  # Вероятности для (grow, prune, change)
            leaf_sd=1.0,  # Стандартное отклонение для значений листьев
            rng_seed=42,
        )
        
        # Сэмплируем из апостериорного распределения
        # MHDecisionTableSampler обрабатывает mu, а для sigma используется автоматический сэмплер
        idata = pm.sample(
            draws=500,  # Количество сэмплов
            tune=500,  # Количество итераций для "прогрева"
            step=[mh_step],
            chains=1,
            random_seed=42,
            return_inferencedata=True,
            compute_convergence_checks=False,  # Отключаем проверки сходимости для ускорения
            progressbar=True,
        )
    
    print("   Сэмплирование завершено")
    
    # ========================================================================
    # 5. Получение предсказаний на тестовой выборке
    # ========================================================================
    print("\n5. Получение предсказаний на тестовой выборке...")
    
    # Для получения предсказаний используем posterior samples
    # Сначала получаем предсказания для mu на тестовых данных
    ppc = None
    with model:
        # Обновляем данные X для предсказания на тестовой выборке
        pm.set_data({"X_data": X_test})
        
        # Получаем предсказания для mu из posterior
        # Используем pm.sample_posterior_predictive для генерации предсказаний
        try:
            ppc = pm.sample_posterior_predictive(
                idata,
                predictions=True,
                extend_inferencedata=True,
                random_seed=42,
            )
            # Извлекаем предсказания
            # Форма: (chains, draws, n_observations)
            y_pred_samples = ppc.predictions["y_obs"].values
            y_pred_mean = np.mean(y_pred_samples, axis=(0, 1))
            print(f"   Получено {y_pred_samples.shape[1]} сэмплов предсказаний")
        except Exception:
            # Альтернативный способ: используем предсказания из posterior для mu
            # и добавляем неопределенность через sigma
            print("   Используем альтернативный способ получения предсказаний...")
            
            # Получаем сэмплы mu и sigma из posterior
            mu_samples = idata.posterior["mu"].values  # (chains, draws, n_obs)
            sigma_samples = idata.posterior["sigma"].values  # (chains, draws)
            
            # Генерируем предсказания с учетом неопределенности
            rng = np.random.default_rng(42)
            n_chains, n_draws, n_obs = mu_samples.shape
            
            y_pred_samples_list = []
            for chain_idx in range(n_chains):
                for draw_idx in range(n_draws):
                    mu = mu_samples[chain_idx, draw_idx, :]
                    sigma = sigma_samples[chain_idx, draw_idx]
                    # Генерируем предсказание с учетом неопределенности
                    y_pred = rng.normal(mu, sigma)
                    y_pred_samples_list.append(y_pred)
            
            y_pred_samples = np.array(y_pred_samples_list)
            # Преобразуем в форму (chains, draws, n_obs)
            y_pred_samples = y_pred_samples.reshape(n_chains, n_draws, n_obs)
            
            # Среднее предсказание
            y_pred_mean = np.mean(y_pred_samples, axis=(0, 1))
            
            print(f"   Получено {y_pred_samples.shape[1]} сэмплов предсказаний (альтернативный способ)")
    
    # ========================================================================
    # 6. Вычисление метрик
    # ========================================================================
    print("\n6. Вычисление метрик на тестовой выборке...")
    
    # RMSE
    rmse = compute_rmse(y_test, y_pred_mean)
    print(f"   RMSE: {rmse:.4f}")
    
    # R²
    r2 = compute_r2(y_test, y_pred_mean)
    print(f"   R²: {r2:.4f}")
    
    # Coverage (95% доверительный интервал)
    # Преобразуем форму: (chains, draws, n_obs) -> (n_samples, n_obs)
    n_chains, n_draws, n_obs = y_pred_samples.shape
    y_pred_flat = y_pred_samples.reshape(n_chains * n_draws, n_obs)
    
    coverage = compute_coverage(y_test, y_pred_flat, alpha=0.05)
    print(f"   Coverage (95% CI): {coverage:.4f} ({coverage*100:.2f}%)")
    
    # ========================================================================
    # 7. Дополнительная информация
    # ========================================================================
    print("\n7. Дополнительная информация:")
    print(f"   Количество decision tables: 50")
    print(f"   Количество сэмплов: {n_draws}")
    print(f"   Количество цепей: {n_chains}")
    
    # Вычисляем ширину доверительного интервала
    lower = np.percentile(y_pred_flat, 2.5, axis=0)
    upper = np.percentile(y_pred_flat, 97.5, axis=0)
    ci_width = np.mean(upper - lower)
    print(f"   Средняя ширина 95% CI: {ci_width:.4f}")
    
    print("\n" + "=" * 60)
    print("Пример завершен успешно!")
    print("=" * 60)
    
    return {
        "rmse": rmse,
        "r2": r2,
        "coverage": coverage,
        "idata": idata,
        "ppc": ppc,
    }


if __name__ == "__main__":
    results = main()
