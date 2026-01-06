# Аппроксимация ЯМР данных
Инструмент для анализа данных ядерно-магнитного резонанса (ЯМР). Программа предназначена для разложения кривых затухания поперечной намагниченности T2, полученных методом CPMG, на сумму экспоненциальных компонент.

Данное программное обеспечение предназначено для количественного анализа данных ядерно-магнитного резонанса (ЯМР). Основная функция приложения — разложение экспериментальных кривых затухания поперечной намагниченности T2 на отдельные экспоненциальные компоненты.

Ссылка на скачивание: https://drive.google.com/drive/folders/18Y2yvi7775m9-ZTp07OsHrJVgGfg6kIe?usp=drive_link

<img width="1920" height="1028" alt="image" src="https://github.com/user-attachments/assets/0dfc2845-836a-4057-a5d4-c94260c32c6e" />



## Методология анализа
В основе вычислительного ядра лежит двухэтапный алгоритм обработки:
  1. Автоматический поиск (NNLS): Метод неотрицательных наименьших квадратов позволяет без участия пользователя определить количество физически значимых компонент и их начальные параметры.
  2. Нелинейная оптимизация (Least Squares): Алгоритм уточняет значения времен релаксации и амплитуд для минимизации ошибки аппроксимации.

## Математическая модель
Программа описывает сигнал 
<img width="879" height="156" alt="image" src="https://github.com/user-attachments/assets/5b6341a8-fc67-43ab-b417-a0cbc51aba53" />


## Описание функционала
Мультимасштабная визуализация: Для детального анализа предусмотрены три типа графиков:
Линейный масштаб — для общей оценки кривой затухания.
Логарифмический масштаб — для анализа быстрорелаксирующих компонент.
График разницы (Residuals) — для визуального контроля точности модели.

## Интерактивный интерфейс:
Возможность ручной корректировки параметров (T₂, доли, смещение) с мгновенным обновлением графиков.
Функция «Копировать из Авто» для быстрой доработки автоматических результатов.
Ручное управление границами осей и толщиной точек.

## Подготовка отчетов: 
Программа формирует итоговое изображение в формате PNG (300 DPI). 
В отчет включаются все типы графиков и таблица с расчетными значениями T2, долями в процентах и параметрами смещения.

## Порядок работы
Загрузка данных: Импорт текстового файла с колонками времени и амплитуды.
Расчёт: Выбор максимального числа компонент и запуск автоматической аппроксимации.
Коррекция: При необходимости — ручная правка параметров на соответствующей вкладке.
Экспорт: Сохранение готового графического отчета для включения в публикацию или отчет по лабораторной работе.


---

# NMR Data Approximation

A software tool for quantitative analysis of Nuclear Magnetic Resonance (NMR) data.  
The application is designed to decompose transverse magnetization decay curves (T₂) obtained using the CPMG method into a sum of exponential components.

This software performs quantitative analysis of NMR relaxation data. The core functionality is the decomposition of experimentally measured transverse relaxation decay curves (T₂) into individual exponential components corresponding to different physical environments.

**Download link:**  
https://drive.google.com/drive/folders/18Y2yvi7775m9-ZTp07OsHrJVgGfg6kIe?usp=drive_link


## Methodology

The computational core is based on a two-stage data processing algorithm:

1. **Automatic component search (NNLS):**  
   The Non-Negative Least Squares method is used to automatically determine the number of physically meaningful components and their initial parameters without user intervention.

2. **Nonlinear optimization (Least Squares):**  
   A nonlinear least-squares algorithm refines the relaxation times and amplitudes to minimize the approximation error.


## Mathematical Model

The measured signal is described by the model:

<img width="879" height="156" alt="Mathematical model" src="https://github.com/user-attachments/assets/5b6341a8-fc67-43ab-b417-a0cbc51aba53" />
Where A is the total amplitude, pi is the fraction of the component, T2 is the relaxation time, and B is the displacement

## Features

### Multi-scale visualization
Three types of plots are provided for detailed analysis:
- **Linear scale** — overall evaluation of the decay curve
- **Logarithmic scale** — analysis of fast-relaxing components
- **Residuals plot** — visual assessment of model accuracy

### Interactive interface
- Manual adjustment of parameters (T₂ values, component fractions, offset) with real-time plot updates
- *Copy from Auto* function for refining automatically obtained results
- Manual control of axis limits and marker size

### Report generation
- Export of a high-resolution report image in PNG format (300 DPI)
- The report includes all plots and a table with calculated T₂ values, component fractions (in percent), and offset parameters


## Workflow

1. **Data loading:** Import a text file containing time and amplitude columns  
2. **Computation:** Select the maximum number of components and run automatic approximation  
3. **Refinement:** Optionally perform manual parameter adjustment  
4. **Export:** Save the final graphical report for use in publications or laboratory reports

