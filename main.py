import sys
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QTableWidget, 
                             QTableWidgetItem, QLabel, QFileDialog, QLineEdit, 
                             QDialog, QFormLayout, QMessageBox, QFrame, QComboBox, QTabWidget,
                             QDoubleSpinBox, QTextEdit, QScrollArea)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from scipy.optimize import nnls, least_squares
import os

# === Стиль графиков "как в Origin" ===
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.linewidth': 1.2,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 5,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'legend.frameon': False,
    'lines.markersize': 4,
    'lines.markeredgewidth': 0.8
})

class NMRCore:
    def fit(self, t, y, max_components=4):
        y_max = np.max(y)
        y_norm = y / y_max
        dt = t[1] - t[0]
        
        t2_grid = np.logspace(np.log10(max(1e-7, dt)), np.log10(t[-1]), 150)
        K = np.exp(-t[:, np.newaxis] / t2_grid)
        amps_grid, _ = nnls(K, y_norm)
        
        peaks = []
        for i in range(1, len(amps_grid) - 1):
            if amps_grid[i] > amps_grid[i-1] and amps_grid[i] > amps_grid[i+1]:
                if amps_grid[i] > np.max(amps_grid) * 0.01:
                    peaks.append([amps_grid[i], t2_grid[i]])
        
        peaks = sorted(peaks, key=lambda x: x[0], reverse=True)[:max_components]
        n = len(peaks)
        if n == 0: return [], y, np.zeros_like(y), 0.0, 1.0
        
        x0 = [p[0] for p in peaks] + [p[1] for p in peaks] + [0.0]
        
        def model_func(p, t_ax):
            n_c = (len(p)-1)//2
            a, t2, off = p[:n_c], p[n_c:2*n_c], p[-1]
            return sum(a[i]*np.exp(-t_ax/t2[i]) for i in range(n_c)) + off

        res = least_squares(lambda p: model_func(p, t) - y_norm, x0, 
                           bounds=([0]*n + [dt/5]*n + [-0.1], [2]*n + [t[-1]*2]*n + [0.1]))
        
        y_f_n = model_func(res.x, t)
        n_c = (len(res.x)-1)//2
        sum_a = np.sum(res.x[:n_c])
        results = [{'T2': res.x[n_c+i], 'Share': res.x[i]/sum_a if sum_a > 0 else 0} for i in range(n_c)]
        
        offset_norm = res.x[-1]
        return sorted(results, key=lambda x: x['T2']), y_f_n * y_max, (y_norm - y_f_n), offset_norm, sum_a

class NMRApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ЯМР Анализатор - Мультиэкспоненциальный подбор")
        self.resize(1200, 750)
        self.core = NMRCore()
        
        self.last_file_path = None
        self.current_t = self.current_y = None
        
        self.auto_fit_res = None
        self.auto_y_fit = None
        self.auto_diff_norm = None
        self.auto_offset_norm = 0.0
        self.auto_amp_scale = 1.0
        
        self.manual_components = []
        self.manual_offset_norm = 0.0
        self.manual_amp_scale = 1.0
        
        self.default_msize = 4.0
        self.reset_graph_settings()
        self.init_ui()

    def reset_graph_settings(self):
        self.graph_settings = [
            {'мин_x': 0, 'макс_x': 1, 'мин_y': 0, 'макс_y': 1.1, 'толщина точек': self.default_msize},
            {'мин_x': 0, 'макс_x': 1, 'мин_y': 1e-4, 'макс_y': 1.1, 'толщина точек': self.default_msize},
            {'мин_x': 0, 'мин_y': -0.05, 'макс_y': 0.05, 'толщина точек': self.default_msize}
        ]

    def show_about_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("О программе и инструкция по использованию")
        dialog.resize(800, 600)
        
        layout = QVBoxLayout(dialog)
        
        text = QTextEdit()
        text.setReadOnly(True)
        text.setStyleSheet("font-size: 11pt; line-height: 1.5;")
        text.setHtml("""
        <h3>ЯМР Анализатор — программа для мультиэкспоненциального анализа кривых релаксации T₂</h3>
        
        <p><b>Назначение</b><br>
        Программа предназначена для обработки данных ядерно-магнитного резонанса (ЯМР) релаксации поперечной намагниченности (T₂), полученных методом CPMG. 
        Основная задача — разложение экспериментальной кривой затухания на дискретный набор экспоненциальных компонент с целью определения времён релаксации T₂ и их относительных вкладов (долей).</p>
        
        <p><b>Методология</b><br>
        Автоматический расчёт основан на комбинации неотрицательного метода наименьших квадратов (NNLS) для предварительного определения спектра T₂ и последующей нелинейной оптимизации (least_squares) для уточнения параметров. 
        Модель имеет вид:</p>
        
        <p style="margin-left: 40px;"><i>S(t) = A × Σ (pᵢ × exp(−t / T₂ᵢ)) + B</i></p>
        
        <p>где:</p>
        <ul>
            <li><b>A</b> — общий масштабный коэффициент (параметр «Амплитуда»);</li>
            <li><b>pᵢ</b> — относительная доля i-й компоненты (Σ pᵢ = 1);</li>
            <li><b>T₂ᵢ</b> — время поперечной релаксации i-й компоненты;</li>
            <li><b>B</b> — постоянное смещение (offset).</li>
        </ul>
        
        <p>Данные нормируются на максимум. Параметр <b>Амплитуда</b> позволяет учесть небольшое отклонение суммы начальных амплитуд от 1.0, что повышает точность подгонки на реальных данных с шумом.</p>
        
        <p><b>Инструкция по использованию</b></p>
        
        <ol>
            <li><b>Загрузка данных</b>: Нажмите «ЗАГРУЗИТЬ ФАЙЛ» и выберите текстовый файл с двумя столбцами (время [с], амплитуда сигнала).</li>
            <li><b>Автоматический расчёт</b>: Перейдите на вкладку «Расчёт», выберите максимальное число компонент и нажмите «РАСЧЕТ». Результаты появятся в таблице, модель отобразится красной линией.</li>
            <li><b>Ручная корректировка</b>: Перейдите на вкладку «Ручной подбор», нажмите «Копировать из Авто», затем редактируйте T₂, доли, смещение или амплитуду. Модель обновится синий линией.</li>
            <li><b>Визуализация</b>: Три графика — линейный, логарифмический и остатки. Нажмите ⚙ для настройки осей.</li>
            <li><b>Экспорт</b>: Нажмите «СОХРАНИТЬ ОТЧЕТ (PNG)» для получения изображения с графиками и таблицей результатов.</li>
        </ol>
        
        <p><b>Перенос результатов в Origin</b><br>
        Используйте формулу:<br>
        <code>y(t) = Амплитуда × Σ (Доляᵢ/100 × exp(−t/T₂ᵢ)) + Смещение_норм × Ymax</code></p>
        
        """)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(text)
        layout.addWidget(scroll)
        
        close_btn = QPushButton("Закрыть")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignRight)
        
        dialog.exec()

    def init_ui(self):
        # Определяем стили для кнопок
        RUN_BUTTON_STYLE = """
            QPushButton {
                background-color: #2ecc71;  /* Яркий зеленый цвет */
                color: white;
                font-weight: bold;
                font-size: 14px;
                border-radius: 8px;
                border: 2px solid #27ae60;
            }
            QPushButton:hover {
                background-color: #27ae60;  /* Темнее при наведении */
            }
            QPushButton:pressed {
                background-color: #1e8449;  /* Еще темнее при нажатии */
                padding-top: 2px; /* Эффект нажатия */
            }
        """

        FILE_BUTTON_STYLE = """
            QPushButton {
                background-color: #3498db;  /* Синий цвет для загрузки */
                color: white;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #2980b9; }
        """

        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        sidebar = QFrame()
        sidebar.setFixedWidth(320)
        sidebar_layout = QVBoxLayout(sidebar)

        sidebar_layout.addWidget(QLabel("<h3>📊 Настройки</h3>"))
        
        self.btn_file = QPushButton("📂 ЗАГРУЗИТЬ ФАЙЛ")
        self.btn_file.setFixedHeight(40)
        self.btn_file.clicked.connect(self.load_file)
        sidebar_layout.addWidget(self.btn_file)

        self.tabs = QTabWidget()
        self.tabs.setMaximumHeight(500)
        sidebar_layout.addWidget(self.tabs)

        nice_table_style = """
            QTableWidget {
                gridline-color: #d3d3d3;
                background-color: #ffffff;
                alternate-background-color: #f9f9f9;
                selection-background-color: #cce5ff;
            }
            QTableWidget::item {
                padding: 6px;
                border-bottom: 1px solid #e0e0e0;
            }
            QHeaderView::section {
                background-color: #f0f0f0;
                padding: 8px;
                border: none;
                border-bottom: 2px solid #c0c0c0;
                font-weight: bold;
                color: #333;
            }
        """

        # === Вкладка "Расчёт" ===
        auto_tab = QWidget()
        auto_layout = QVBoxLayout(auto_tab)
        
        auto_layout.addWidget(QLabel("<b>Количество компонент:</b>"))
        self.comp_box = QComboBox()
        self.comp_box.addItems(["1", "2", "3", "4", "5", "6"])
        self.comp_box.setCurrentIndex(3)
        auto_layout.addWidget(self.comp_box)
        
        hint = QLabel("<i>Ограничивает максимальное число экспонент.\nРекомендуется 4 для большинства данных.</i>")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #555; font-size: 10px;")
        auto_layout.addWidget(hint)

        self.btn_run = QPushButton("🚀 РАСЧЕТ")
        self.btn_run.setFixedHeight(45)
        self.btn_run.setStyleSheet(RUN_BUTTON_STYLE)
        self.btn_run.clicked.connect(self.run_auto_calc)
        auto_layout.addWidget(self.btn_run)
        
        self.auto_table = QTableWidget(0, 2)
        self.auto_table.setHorizontalHeaderLabels(["T2 (сек)", "Доля (%)"])
        self.auto_table.horizontalHeader().setStretchLastSection(True)
        self.auto_table.setStyleSheet(nice_table_style)
        auto_layout.addWidget(self.auto_table)
        
        self.auto_params_table = QTableWidget(2, 2)
        self.auto_params_table.setHorizontalHeaderLabels(["Параметр", "Значение (норм. / абс.)"])
        self.auto_params_table.horizontalHeader().setStretchLastSection(True)
        self.auto_params_table.setStyleSheet(nice_table_style)
        self.auto_params_table.verticalHeader().setVisible(False)
        self.auto_params_table.setMaximumHeight(100)
        self.auto_params_table.setItem(0, 0, QTableWidgetItem("Смещение"))
        self.auto_params_table.setItem(1, 0, QTableWidgetItem("Амплитуда"))
        self.auto_params_table.setItem(0, 1, QTableWidgetItem("0.0000 / 0.00"))
        self.auto_params_table.setItem(1, 1, QTableWidgetItem("1.0000"))
        auto_layout.addWidget(self.auto_params_table)
        
        self.tabs.addTab(auto_tab, "Расчёт")

        # === Вкладка "Ручной подбор" ===
        manual_tab = QWidget()
        manual_layout = QVBoxLayout(manual_tab)
        
        manual_layout.addWidget(QLabel("<b>Ручная аппроксимация:</b>"))
        
        self.btn_copy_from_auto = QPushButton("← Копировать из Авто")
        self.btn_copy_from_auto.setFixedHeight(35)
        self.btn_copy_from_auto.clicked.connect(self.copy_auto_to_manual)
        manual_layout.addWidget(self.btn_copy_from_auto)
        
        self.manual_table = QTableWidget(0, 2)
        self.manual_table.setHorizontalHeaderLabels(["T2 (сек)", "Доля (%)"])
        self.manual_table.horizontalHeader().setStretchLastSection(True)
        self.manual_table.itemChanged.connect(self.manual_update)
        self.manual_table.setStyleSheet(nice_table_style)
        manual_layout.addWidget(self.manual_table)
        
        tools = QHBoxLayout()
        self.btn_add = QPushButton("+ Добавить")
        self.btn_add.clicked.connect(self.add_manual_component)
        self.btn_del = QPushButton("- Удалить")
        self.btn_del.clicked.connect(self.remove_manual_component)
        tools.addWidget(self.btn_add)
        tools.addWidget(self.btn_del)
        manual_layout.addLayout(tools)
        
        self.manual_params_table = QTableWidget(2, 2)
        self.manual_params_table.setHorizontalHeaderLabels(["Параметр", "Значение (норм. / абс.)"])
        self.manual_params_table.horizontalHeader().setStretchLastSection(True)
        self.manual_params_table.setStyleSheet(nice_table_style)
        self.manual_params_table.verticalHeader().setVisible(False)
        self.manual_params_table.setMaximumHeight(100)
        self.manual_params_table.setItem(0, 0, QTableWidgetItem("Смещение"))
        self.manual_params_table.setItem(1, 0, QTableWidgetItem("Амплитуда"))
        self.manual_params_table.itemChanged.connect(self.manual_update)
        self.manual_params_table.setItem(0, 1, QTableWidgetItem("0.0000 / 0.00"))
        self.manual_params_table.setItem(1, 1, QTableWidgetItem("1.0000"))
        manual_layout.addWidget(self.manual_params_table)
        
        self.manual_status = QLabel("Добавьте компоненты или скопируйте из авто")
        self.manual_status.setStyleSheet("color: #666;")
        manual_layout.addWidget(self.manual_status)
        
        self.tabs.addTab(manual_tab, "Ручной подбор")

        # Нижние кнопки в боковой панели
        self.btn_save = QPushButton("💾 СОХРАНИТЬ ОТЧЕТ (PNG)")
        self.btn_save.clicked.connect(self.save_report_png)
        sidebar_layout.addWidget(self.btn_save)
        
        self.btn_about = QPushButton("ℹ️ О программе и инструкция")
        self.btn_about.setFixedHeight(35)
        self.btn_about.clicked.connect(self.show_about_dialog)
        sidebar_layout.addWidget(self.btn_about)
        
        self.status_label = QLabel("Ожидание файла...")
        sidebar_layout.addWidget(self.status_label)
        sidebar_layout.addStretch()

        main_layout.addWidget(sidebar)

        # === Графики (только стандартные три) ===
        self.figs, self.canvases, self.axes = [], [], []
        titles = ["Линейный масштаб (норм.)", "Логарифмический масштаб", "Разница"]

        graphs_widget = QWidget()
        graphs_layout = QHBoxLayout(graphs_widget)
        col_left = QVBoxLayout()
        col_right = QVBoxLayout()

        for i in range(3):
            fig = Figure(tight_layout=True)
            canvas = FigureCanvas(fig)
            self.figs.append(fig)
            self.canvases.append(canvas)
            self.axes.append(fig.add_subplot(111))
            
            header_w, header_l = self.create_plot_header(i, titles[i])
            header_l.addWidget(canvas)
            if i < 2: 
                col_left.addWidget(header_w)
            else: 
                col_right.addWidget(header_w)

        graphs_layout.addLayout(col_left, 2)
        graphs_layout.addLayout(col_right, 3)

        main_layout.addWidget(graphs_widget)

    def create_plot_header(self, index, title):
        container = QWidget()
        layout = QVBoxLayout(container)
        header = QHBoxLayout()
        header.addWidget(QLabel(f"<b>{title}</b>"))
        if index < 3:
            btn_s = QPushButton("⚙")
            btn_s.setFixedSize(22, 22)
            btn_s.clicked.connect(lambda: self.open_settings(index))
            header.addWidget(btn_s)
        header.addStretch()
        layout.addLayout(header)
        return container, layout

    def load_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Открыть данные", "", "Данные (*.txt *.nmr)")
        if not path: return
        try:
            if path != self.last_file_path:
                self.reset_graph_settings()
                self.last_file_path = path

            data = np.loadtxt(path)
            self.current_t, self.current_y = data[:, 0], data[:, 1]
            if self.current_t[0] > 10: 
                self.current_t /= 1e6

            t_end = self.current_t[-1]
            for i in range(3): 
                self.graph_settings[i]['макс_x'] = t_end

            self.auto_fit_res = self.auto_y_fit = self.auto_diff_norm = None
            self.auto_offset_norm = 0.0
            self.auto_amp_scale = 1.0
            self.auto_table.setRowCount(0)
            self.auto_params_table.setItem(0, 1, QTableWidgetItem("0.0000 / 0.00"))
            self.auto_params_table.setItem(1, 1, QTableWidgetItem("1.0000"))
            
            self.manual_components = []
            self.manual_offset_norm = 0.0
            self.manual_amp_scale = 1.0
            self.manual_table.setRowCount(0)
            self.manual_params_table.setItem(0, 1, QTableWidgetItem("0.0000 / 0.00"))
            self.manual_params_table.setItem(1, 1, QTableWidgetItem("1.0000"))

            self.status_label.setText("Файл загружен")
            self.draw()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить файл:\n{str(e)}")

    def run_auto_calc(self):
        if self.current_t is None: return
        try:
            n = int(self.comp_box.currentText())
            self.auto_fit_res, self.auto_y_fit, diff_raw, self.auto_offset_norm, self.auto_amp_scale = self.core.fit(self.current_t, self.current_y, n)
            self.auto_diff_norm = diff_raw

            self.auto_table.setRowCount(len(self.auto_fit_res))
            for i, r in enumerate(self.auto_fit_res):
                self.auto_table.setItem(i, 0, QTableWidgetItem(f"{r['T2']:.3f}"))
                self.auto_table.setItem(i, 1, QTableWidgetItem(f"{r['Share']*100:.2f}"))

            y_max = np.max(self.current_y)
            offset_abs = self.auto_offset_norm * y_max
            self.auto_params_table.setItem(0, 1, QTableWidgetItem(f"{self.auto_offset_norm:.4f} / {offset_abs:.2f}"))
            self.auto_params_table.setItem(1, 1, QTableWidgetItem(f"{self.auto_amp_scale:.4f}"))

            self.status_label.setText(f"Авто: {len(self.auto_fit_res)} компонент")
            self.draw()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка расчёта", str(e))

    def copy_auto_to_manual(self):
        if not self.auto_fit_res:
            QMessageBox.information(self, "Нет данных", "Сначала выполните автоматический расчёт")
            return
        
        self.manual_table.blockSignals(True)
        self.manual_table.setRowCount(len(self.auto_fit_res))
        
        for i, r in enumerate(self.auto_fit_res):
            self.manual_table.setItem(i, 0, QTableWidgetItem(f"{r['T2']:.3f}"))
            self.manual_table.setItem(i, 1, QTableWidgetItem(f"{r['Share']*100:.2f}"))

        y_max = np.max(self.current_y)
        offset_abs = self.auto_offset_norm * y_max
        self.manual_params_table.setItem(0, 1, QTableWidgetItem(f"{self.auto_offset_norm:.4f} / {offset_abs:.2f}"))
        self.manual_params_table.setItem(1, 1, QTableWidgetItem(f"{self.auto_amp_scale:.4f}"))
        
        self.manual_offset_norm = self.auto_offset_norm
        self.manual_amp_scale = self.auto_amp_scale
        
        self.manual_table.blockSignals(False)
        self.manual_update()

    def add_manual_component(self):
        row = self.manual_table.rowCount()
        self.manual_table.insertRow(row)
        self.manual_table.setItem(row, 0, QTableWidgetItem("0.100"))
        self.manual_table.setItem(row, 1, QTableWidgetItem("25.00"))
        self.manual_update()

    def remove_manual_component(self):
        row = self.manual_table.currentRow()
        if row >= 0:
            self.manual_table.removeRow(row)
            self.manual_update()

    def manual_update(self):
        if self.current_t is None: return
        
        try:
            self.manual_components = []
            total_percent = 0.0
            valid = True
            
            y_max = np.max(self.current_y)
            
            for i in range(self.manual_table.rowCount()):
                t2_item = self.manual_table.item(i, 0)
                share_item = self.manual_table.item(i, 1)
                if not t2_item or not share_item:
                    continue
                
                t2_text = t2_item.text().strip().replace(',', '.')
                share_text = share_item.text().strip().replace(',', '.')
                
                try:
                    t2 = float(t2_text)
                    share_percent = float(share_text)
                    if t2 <= 0 or share_percent < 0:
                        valid = False
                        break
                    self.manual_components.append({'T2': t2, 'Share': share_percent / 100.0})
                    total_percent += share_percent
                except ValueError:
                    valid = False
                    break
            
            offset_item = self.manual_params_table.item(0, 1)
            if offset_item:
                text = offset_item.text().split('/')[0].strip().replace(',', '.')
                try:
                    self.manual_offset_norm = float(text)
                except:
                    self.manual_offset_norm = 0.0
            
            amp_item = self.manual_params_table.item(1, 1)
            if amp_item:
                text = amp_item.text().strip().replace(',', '.')
                try:
                    self.manual_amp_scale = float(text)
                except:
                    self.manual_amp_scale = 1.0
            
            offset_abs = self.manual_offset_norm * y_max
            self.manual_params_table.blockSignals(True)
            self.manual_params_table.setItem(0, 1, QTableWidgetItem(f"{self.manual_offset_norm:.4f} / {offset_abs:.2f}"))
            self.manual_params_table.setItem(1, 1, QTableWidgetItem(f"{self.manual_amp_scale:.4f}"))
            self.manual_params_table.blockSignals(False)
            
            if valid and self.manual_components:
                self.manual_status.setText(f"Ручной: {len(self.manual_components)} комп. | Сумма: {total_percent:.1f}%")
            else:
                self.manual_status.setText("Некорректные значения")
                self.manual_components = []
            
            self.draw()
        except:
            self.manual_status.setText("Ошибка ввода")
            self.manual_components = []

    def draw(self):
        if self.current_t is None: return
        
        y_max = np.max(self.current_y)
        y_norm = self.current_y / y_max

        auto_model_norm = self.auto_y_fit / y_max if self.auto_y_fit is not None else None
        auto_diff = self.auto_diff_norm if self.auto_diff_norm is not None else None

        manual_model_norm = None
        manual_diff = None
        if self.manual_components:
            total_share = sum(c['Share'] for c in self.manual_components)
            if total_share > 0:
                exp_sum = sum((c['Share'] / total_share) * np.exp(-self.current_t / c['T2']) for c in self.manual_components)
                manual_model_norm = self.manual_amp_scale * exp_sum + self.manual_offset_norm
                manual_diff = y_norm - manual_model_norm

        for i, ax in enumerate(self.axes):
            ax.clear()
            s = self.graph_settings[i]
            ms = s['толщина точек']
            
            ax.set_xlabel("Время (с)", fontsize=11, labelpad=8)

            if i == 0:
                ax.plot(self.current_t, y_norm, 'k.', ms=ms, label='Данные')
                if auto_model_norm is not None:
                    ax.plot(self.current_t, auto_model_norm, 'r-', lw=1.8, label='Авто')
                if manual_model_norm is not None:
                    ax.plot(self.current_t, manual_model_norm, 'b-', lw=1.8, label='Ручной')
                ax.set_ylabel("Амплитуда сигнала (норм.)", fontsize=11, labelpad=8)
                if auto_model_norm is not None or manual_model_norm is not None:
                    ax.legend(fontsize=9)
            elif i == 1:
                ax.semilogy(self.current_t, y_norm, 'k.', ms=ms)
                if auto_model_norm is not None:
                    ax.semilogy(self.current_t, auto_model_norm, 'r-', lw=1.8)
                if manual_model_norm is not None:
                    ax.semilogy(self.current_t, manual_model_norm, 'b-', lw=1.8)
                ax.set_ylabel("Амплитуда сигнала (норм.)", fontsize=11, labelpad=8)
            else:
                if auto_diff is not None:
                    ax.plot(self.current_t, auto_diff, 'r.', ms=ms, label='Авто')
                if manual_diff is not None:
                    ax.plot(self.current_t, manual_diff, 'b.', ms=ms, label='Ручной')
                ax.set_ylabel("Разница (норм.)", fontsize=11, labelpad=8)
                ax.axhline(0, color='gray', lw=1, ls='--')
                if auto_diff is not None or manual_diff is not None:
                    ax.legend(fontsize=9)
            
            ax.set_xlim(s['мин_x'], s['макс_x'])
            ax.set_ylim(s['мин_y'], s['макс_y'])
            self.canvases[i].draw_idle()

    def open_settings(self, idx):
        dialog = QDialog(self)
        dialog.setWindowTitle("Настройки осей")
        form = QFormLayout(dialog)
        edits = {}
        for k, v in self.graph_settings[idx].items():
            edits[k] = QLineEdit(str(v))
            form.addRow(f"<b>{k}</b>:", edits[k])
        btn = QPushButton("Применить")
        btn.clicked.connect(dialog.accept)
        form.addRow(btn)
        
        if dialog.exec():
            try:
                for k in edits:
                    self.graph_settings[idx][k] = float(edits[k].text().replace(',', '.'))
                self.draw()
            except:
                pass

    def save_report_png(self):
        if self.auto_fit_res is None:
            QMessageBox.information(self, "Инфо", "Для отчёта выполните автоматический расчёт")
            return
        
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить отчет", "ЯМР_Отчет.png", "PNG (*.png)")
        if not path: return

        y_max = np.max(self.current_y)
        report_fig = plt.figure(figsize=(14, 10))
        gs = report_fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        ax1 = report_fig.add_subplot(gs[0, 0])
        ax1.plot(self.current_t, self.current_y/y_max, 'k.', label='Данные')
        ax1.plot(self.current_t, self.auto_y_fit/y_max, 'r-', label='Модель (авто)')
        ax1.set_title("Линейный масштаб (норм.)")
        ax1.set_xlabel("Время (с)", fontsize=11)
        ax1.set_ylabel("Амплитуда сигнала (норм.)", fontsize=11)
        ax1.legend()

        ax2 = report_fig.add_subplot(gs[1, 0])
        ax2.semilogy(self.current_t, self.current_y/y_max, 'k.')
        ax2.semilogy(self.current_t, self.auto_y_fit/y_max, 'r-')
        ax2.set_title("Логарифмический масштаб")
        ax2.set_xlabel("Время (с)", fontsize=11)
        ax2.set_ylabel("Амплитуда сигнала (норм.)", fontsize=11)

        ax3 = report_fig.add_subplot(gs[0, 1])
        ax3.plot(self.current_t, self.auto_diff_norm, 'g.')
        ax3.axhline(0, color='red', ls='--')
        ax3.set_title("Разница")
        ax3.set_xlabel("Время (с)", fontsize=11)
        ax3.set_ylabel("Разница (норм.)", fontsize=11)

        ax4 = report_fig.add_subplot(gs[1, 1])
        ax4.axis('off')
        
        cell_text = [[f"{r['T2']:.3f}", f"{r['Share']*100:.2f}%"] for r in self.auto_fit_res]
        
        offset_abs = self.auto_offset_norm * y_max
        cell_text.append(["Смещение", f"{self.auto_offset_norm:.4f} / {offset_abs:.2f}"])
        cell_text.append(["Амплитуда", f"{self.auto_amp_scale:.4f}"])
        
        full_cell_text = [["Таблица результатов", ""]] + \
                         [["T2 (сек)", "Доля (%)"]] + \
                         cell_text

        table = ax4.table(cellText=full_cell_text, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.2)

        for i in range(2):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='w')
        for i in range(2):
            table[(1, i)].set_facecolor('#aaaaaa')
            table[(1, i)].set_text_props(weight='bold', color='k')

        report_fig.tight_layout(pad=1.0)
        report_fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(report_fig)
        QMessageBox.information(self, "Успех", "Отчет успешно сохранен!")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NMRApp()
    window.show()
    sys.exit(app.exec())