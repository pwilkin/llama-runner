from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QTextEdit, QDialogButtonBox, QSizePolicy, QPushButton
)
from PySide6.QtCore import QTimer


class LogViewerDialog(QDialog):
    """
    Custom dialog to display live logs from a running process.
    """
    def __init__(self, title, log_provider_callback, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(800)  # Wide enough for standard console lines
        self.setMinimumHeight(400)
        
        # Callback to get logs
        self.log_provider_callback = log_provider_callback
        
        layout = QVBoxLayout()
        
        # Create text edit for logs
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        self.log_text_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.log_text_edit)
        
        # Add refresh button
        button_layout = QDialogButtonBox()
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_logs)
        button_layout.addButton(self.refresh_button, QDialogButtonBox.ButtonRole.ActionRole)
        
        # Add auto-refresh checkbox
        self.auto_refresh_button = QPushButton("Auto-refresh")
        self.auto_refresh_button.setCheckable(True)
        self.auto_refresh_button.clicked.connect(self.toggle_auto_refresh)
        button_layout.addButton(self.auto_refresh_button, QDialogButtonBox.ButtonRole.ActionRole)
        
        # Add close button
        close_button = button_layout.addButton(QDialogButtonBox.StandardButton.Close)
        close_button.clicked.connect(self.accept)
        
        layout.addWidget(button_layout)
        
        self.setLayout(layout)
        
        # Timer for auto-refresh
        self.auto_refresh_timer = QTimer(self)
        self.auto_refresh_timer.timeout.connect(self.refresh_logs)
        
        # Initial log load
        self.refresh_logs()
    
    def refresh_logs(self):
        """Refresh the log display with current logs."""
        try:
            log_lines = self.log_provider_callback()
            if log_lines:
                self.log_text_edit.setPlainText("\n".join(log_lines))
                # Scroll to bottom to show latest logs
                scrollbar = self.log_text_edit.verticalScrollBar()
                scrollbar.setValue(scrollbar.maximum())
            else:
                self.log_text_edit.setPlainText("No logs available.")
        except Exception as e:
            self.log_text_edit.setPlainText(f"Error retrieving logs: {str(e)}")
    
    def toggle_auto_refresh(self):
        """Toggle auto-refresh functionality."""
        if self.auto_refresh_button.isChecked():
            self.auto_refresh_button.setText("Auto-refresh ON")
            self.auto_refresh_timer.start(1000)  # Refresh every second
        else:
            self.auto_refresh_button.setText("Auto-refresh")
            self.auto_refresh_timer.stop()
    
    def closeEvent(self, event):
        """Stop timer when dialog is closed."""
        self.auto_refresh_timer.stop()
        super().closeEvent(event)