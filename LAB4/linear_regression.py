from __future__ import annotations

from typing import List

import numpy as np

from descents import BaseDescent
from descents import get_descent


class LinearRegression:
    """
    Класс линейной регрессии.

    Parameters
    ----------
    descent_config : dict
        Конфигурация градиентного спуска.
    tolerance : float, optional
        Критерий остановки для квадрата евклидова нормы разности весов. По умолчанию равен 1e-4.
    max_iter : int, optional
        Критерий остановки по количеству итераций. По умолчанию равен 300.

    Attributes
    ----------
    descent : BaseDescent
        Экземпляр класса, реализующего градиентный спуск.
    tolerance : float
        Критерий остановки для квадрата евклидова нормы разности весов.
    max_iter : int
        Критерий остановки по количеству итераций.
    loss_history : List[float]
        История значений функции потерь на каждой итерации.

    """

    def __init__(self, descent_config: dict, tolerance: float = 1e-4, max_iter: int = 300):
        """
        :param descent_config: gradient descent config
        :param tolerance: stopping criterion for square of euclidean norm of weight difference (float)
        :param max_iter: stopping criterion for iterations (int)
        """
        self.descent: BaseDescent = get_descent(descent_config)

        self.tolerance: float = tolerance
        self.max_iter: int = max_iter

        self.loss_history: List[float] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> LinearRegression:
        """
        Обучение модели линейной регрессии, подбор весов для наборов данных x и y.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.
        y : np.ndarray
            Массив целевых переменных.

        Returns
        -------
        self : LinearRegression
            Возвращает экземпляр класса с обученными весами.

        """
        # TODO: реализовать подбор весов для x и y

        # Начальные веса устанавливаются случайным образом
        weights = np.random.randn(x.shape[1])

        # Сохраняем начальное значение функции потерь перед началом обучения
        initial_loss = self.calc_loss(x, y)
        self.loss_history.append(initial_loss)

        for i in range(self.max_iter):
            # Вычисляем градиенты для текущего вектора весов
            gradients = self.descent.calc_gradient(x, y)

            # Обновляем веса
            new_weights = self.descent.update_weights(gradients)

            # Проверка условия остановки по наличию NaN в весах
            if np.isnan(new_weights).any():
                print("Вес содержит NaN, обучение остановлено.")
                break

            # Вычисление разницы между текущими и новыми весами
            diff_norm = np.linalg.norm(new_weights - weights)

            # Проверка условия остановки по норме разности весов
            if diff_norm <= self.tolerance:
                print(f"Евклидова норма разности весов достигла {diff_norm:.6f}, обучение остановлено.")
                break

            # Обновляем веса
            weights = new_weights

            # Запись нового значения функции потерь
            current_loss = self.calc_loss(x, y)
            self.loss_history.append(current_loss)

        # После завершения обучения сохраняем финальные веса
        self.descent.w = weights

        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Прогнозирование целевых переменных для набора данных x.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.

        Returns
        -------
        prediction : np.ndarray
            Массив прогнозируемых значений.
        """
        return self.descent.predict(x)

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Расчёт значения функции потерь для наборов данных x и y.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.
        y : np.ndarray
            Массив целевых переменных.

        Returns
        -------
        loss : float
            Значение функции потерь.
        """
        return self.descent.calc_loss(x, y)