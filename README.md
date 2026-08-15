# NMR T₂ Relaxation Analyzer

[![Platform](https://img.shields.io/badge/Platform-Windows-0078D6)](https://www.microsoft.com/windows/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![GUI](https://img.shields.io/badge/GUI-PyQt6-41CD52)](https://www.riverbankcomputing.com/software/pyqt/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Программа для мультиэкспоненциального анализа кривых поперечной релаксации ЯМР.

Desktop application for multi-exponential analysis of NMR transverse relaxation curves.

[Русская версия](#русский) · [English version](#english)

---

## Русский

## Скачать готовую программу

### [Скачать NMR T₂ Relaxation Analyzer для Windows](https://drive.google.com/drive/folders/18Y2yvi7775m9-ZTp07OsHrJVgGfg6kIe?usp=drive_link)

Для использования программы не требуется устанавливать Python или самостоятельно запускать исходный код.

### Запуск программы

1. Скачайте архив с Google Drive.
2. Распакуйте архив в любую папку.
3. Откройте распакованную папку `dist`.
4. Запустите находящийся внутри `.exe`-файл.
5. При необходимости создайте ярлык приложения на рабочем столе.

> Не перемещайте `.exe`-файл отдельно от остальных файлов папки `dist`: они могут быть необходимы для работы приложения.

## Быстрый старт

1. Нажмите **«Загрузить файл»**.
2. Выберите файл с данными в формате `.txt` или `.nmr`.
3. Укажите максимальное количество экспоненциальных компонент.
4. Нажмите **«Расчёт»**.
5. Проверьте найденные значения T₂, доли компонент и график остатков.
6. При необходимости скопируйте результат во вкладку ручного подбора и скорректируйте параметры.
7. Нажмите **«Сохранить отчёт»**, чтобы получить итоговое изображение в PNG.

## О программе

**NMR T₂ Relaxation Analyzer** — настольное приложение для количественного анализа данных ядерно-магнитного резонанса.

Программа разлагает экспериментальную кривую затухания поперечной намагниченности, полученную методом CPMG, на сумму экспоненциальных компонент и определяет:

- времена релаксации T₂;
- относительную долю каждой компоненты;
- постоянное смещение сигнала;
- суммарную амплитуду модели;
- остатки аппроксимации.

Приложение объединяет автоматический расчёт, ручную корректировку параметров и создание готового графического отчёта.

## Основные возможности

- автоматический поиск значимых компонент;
- выбор максимального количества компонент от 1 до 6;
- расчёт времён релаксации T₂ и относительных долей;
- ручное добавление и удаление компонент;
- перенос результатов автоматического расчёта в ручной режим;
- редактирование T₂, долей, амплитуды и смещения;
- линейный и логарифмический графики;
- визуализация остатков аппроксимации;
- настройка границ осей и размера маркеров;
- экспорт отчёта в PNG с разрешением 300 DPI.

## Формат входных данных

Программа принимает файлы `.txt` и `.nmr`.

Файл должен содержать минимум два числовых столбца без заголовка:

```text
0.001  1.0000
0.002  0.9431
0.003  0.8914
0.004  0.8447
0.005  0.8021
```

- первый столбец — время;
- второй столбец — амплитуда сигнала;
- разделитель — пробел или табуляция.

Рекомендуется передавать время в секундах. Если первое значение времени больше 10, текущая версия программы считает значения микросекундами и делит их на `1 000 000`.

---

## English

## Download the Windows Application

### [Download NMR T₂ Relaxation Analyzer](https://drive.google.com/drive/folders/18Y2yvi7775m9-ZTp07OsHrJVgGfg6kIe?usp=drive_link)

Python is not required to use the compiled Windows application.

### Running the Application

1. Download the archive from Google Drive.
2. Extract it into any directory.
3. Open the extracted `dist` directory.
4. Run the included `.exe` file.
5. Optionally create a desktop shortcut.

> Keep the executable inside the `dist` directory. Other files in this directory may be required by the application.

## Quick Start

1. Click **Load File**.
2. Select a `.txt` or `.nmr` data file.
3. Choose the maximum number of exponential components.
4. Run the automatic calculation.
5. Inspect the estimated T₂ values, component fractions and residuals.
6. If necessary, copy the result into manual mode and adjust the parameters.
7. Export the final report as a PNG image.

## About

**NMR T₂ Relaxation Analyzer** is a desktop application for quantitative analysis of Nuclear Magnetic Resonance data.

It decomposes transverse magnetization decay curves acquired using the CPMG method into a sum of exponential components and estimates:

- T₂ relaxation times;
- relative component fractions;
- constant signal offset;
- total model amplitude;
- fitting residuals.

## Features

- automatic component detection;
- configurable maximum of 1–6 components;
- T₂ and component-fraction estimation;
- manual component editing;
- transfer of automatic results into manual mode;
- editable T₂ values, fractions, amplitude and offset;
- linear and logarithmic visualizations;
- residual analysis;
- configurable axes and marker size;
- 300 DPI PNG report generation.

## Input Format

The application accepts `.txt` and `.nmr` files containing at least two numeric columns without a header:

```text
0.001  1.0000
0.002  0.9431
0.003  0.8914
0.004  0.8447
0.005  0.8021
```

- first column — time;
- second column — signal amplitude;
- separator — spaces or tabs.

Time values in seconds are recommended. If the first time value is greater than 10, the current implementation interprets the values as microseconds and divides them by `1,000,000`.
