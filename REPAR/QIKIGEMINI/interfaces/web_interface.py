#!/usr/bin/env python3
"""
QIKI Web Interface - Современный веб-интерфейс для управления симуляцией
=======================================================================
"""

from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_socketio import SocketIO, emit
import json
import threading
import time
import numpy as np
from typing import Dict, Any, Optional
import logging
import os


class QIKIWebInterface:
    """
    Веб-интерфейс для QIKI симуляции
    
    Возможности:
    - Мониторинг в реальном времени
    - Управление параметрами симуляции
    - Визуализация данных
    - Контроль ИИ агента
    """
    
    def __init__(self, qiki_simulation=None, host='127.0.0.1', port=8080):
        self.app = Flask(__name__, 
                        template_folder='web/templates',
                        static_folder='web/static')
        self.app.config['SECRET_KEY'] = 'qiki_secret_key_2024'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        self.simulation = qiki_simulation
        self.host = host
        self.port = port
        self.running = False
        
        # Данные для передачи клиентам
        self.current_data = {
            "status": "initializing",
            "position": [0, 0, 0],
            "velocity": [0, 0, 0],
            "target": [10, 10, 10],
            "battery": 100.0,
            "temperature": 20.0,
            "integrity": 100.0,
            "ai_mode": "autonomous",
            "timestamp": time.time()
        }
        
        self._setup_routes()
        self._setup_socketio()
        
        logging.getLogger('werkzeug').setLevel(logging.WARNING)
        
    def _setup_routes(self):
        """Настройка HTTP маршрутов"""
        
        @self.app.route('/')
        def index():
            """Главная страница"""
            return render_template('dashboard.html')
        
        @self.app.route('/api/status')
        def api_status():
            """API статуса симуляции"""
            return jsonify(self.current_data)
        
        @self.app.route('/api/control', methods=['POST'])
        def api_control():
            """API управления симуляцией"""
            try:
                data = request.get_json()
                command = data.get('command')
                params = data.get('parameters', {})
                
                if command == 'set_target':
                    target = params.get('target', [10, 10, 10])
                    if self.simulation and hasattr(self.simulation, 'agent'):
                        self.simulation.agent.set_target(np.array(target))
                    return jsonify({"status": "success", "message": f"Target set to {target}"})
                
                elif command == 'pause':
                    if self.simulation:
                        self.simulation._paused = True
                    return jsonify({"status": "success", "message": "Simulation paused"})
                
                elif command == 'resume':
                    if self.simulation:
                        self.simulation._paused = False
                    return jsonify({"status": "success", "message": "Simulation resumed"})
                
                elif command == 'emergency_stop':
                    if self.simulation and hasattr(self.simulation, 'agent'):
                        self.simulation.agent.emergency_stop = True
                    return jsonify({"status": "success", "message": "Emergency stop activated"})
                
                elif command == 'ai_mode':
                    mode = params.get('mode', 'autonomous')
                    # TODO: Реализовать смену режима ИИ
                    return jsonify({"status": "success", "message": f"AI mode set to {mode}"})
                
                else:
                    return jsonify({"status": "error", "message": f"Unknown command: {command}"})
                    
            except Exception as e:
                return jsonify({"status": "error", "message": str(e)})
        
        @self.app.route('/api/data/export')
        def api_export_data():
            """Экспорт данных симуляции"""
            try:
                # TODO: Реализовать экспорт данных
                return jsonify({
                    "status": "success", 
                    "message": "Data export not implemented yet"
                })
            except Exception as e:
                return jsonify({"status": "error", "message": str(e)})
    
    def _setup_socketio(self):
        """Настройка WebSocket соединений"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Обработка подключения клиента"""
            emit('status', self.current_data)
            print(f"Client connected: {request.sid}")
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Обработка отключения клиента"""
            print(f"Client disconnected: {request.sid}")
        
        @self.socketio.on('command')
        def handle_command(data):
            """Обработка команд от клиента"""
            try:
                command = data.get('command')
                params = data.get('parameters', {})
                
                if command == 'set_target':
                    target = params.get('target', [10, 10, 10])
                    if self.simulation and hasattr(self.simulation, 'agent'):
                        self.simulation.agent.set_target(np.array(target))
                    emit('response', {
                        "status": "success", 
                        "message": f"Target set to {target}"
                    })
                
                # Добавить другие команды по мере необходимости
                
            except Exception as e:
                emit('response', {"status": "error", "message": str(e)})
    
    def update_data(self, simulation_data: Dict[str, Any]):
        """Обновление данных для передачи клиентам"""
        self.current_data.update(simulation_data)
        self.current_data['timestamp'] = time.time()
        
        # Отправка данных всем подключенным клиентам
        if self.running:
            self.socketio.emit('update', self.current_data)
    
    def start_server(self):
        """Запуск веб-сервера"""
        self.running = True
        
        # Создание директорий для статики и шаблонов если не существуют
        os.makedirs('web/templates', exist_ok=True)
        os.makedirs('web/static', exist_ok=True)
        
        # Создание базового HTML шаблона если не существует
        self._create_default_template()
        
        print(f"🌐 Starting QIKI Web Interface on http://{self.host}:{self.port}")
        
        try:
            self.socketio.run(
                self.app,
                host=self.host,
                port=self.port,
                debug=False,
                allow_unsafe_werkzeug=True
            )
        except Exception as e:
            print(f"❌ Failed to start web server: {e}")
    
    def stop_server(self):
        """Остановка веб-сервера"""
        self.running = False
    
    def _create_default_template(self):
        """Создание базового HTML шаблона"""
        template_path = 'web/templates/dashboard.html'
        
        if not os.path.exists(template_path):
            html_content = '''<!DOCTYPE html>
<html>
<head>
    <title>QIKI Control Dashboard</title>
    <meta charset="UTF-8">
    <script src="https://cdn.socket.io/4.0.0/socket.io.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 30px; }
        .status-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .status-card { 
            background: rgba(255,255,255,0.1); 
            padding: 20px; 
            border-radius: 10px; 
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        .control-panel { 
            background: rgba(255,255,255,0.1); 
            padding: 20px; 
            border-radius: 10px; 
            margin-bottom: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        .btn { 
            background: #4CAF50; 
            color: white; 
            padding: 10px 20px; 
            border: none; 
            border-radius: 5px; 
            cursor: pointer; 
            margin: 5px;
            transition: background 0.3s;
        }
        .btn:hover { background: #45a049; }
        .btn.danger { background: #f44336; }
        .btn.danger:hover { background: #da190b; }
        .input-group { margin: 10px 0; }
        .input-group label { display: block; margin-bottom: 5px; }
        .input-group input { 
            width: 100%; 
            padding: 8px; 
            border: none; 
            border-radius: 4px; 
            background: rgba(255,255,255,0.2);
            color: white;
        }
        .input-group input::placeholder { color: rgba(255,255,255,0.7); }
        #plot { height: 400px; margin-top: 20px; }
        .status { font-size: 14px; color: #ccc; }
        .value { font-size: 24px; font-weight: bold; }
        .connection-status { 
            position: fixed; 
            top: 10px; 
            right: 10px; 
            padding: 5px 10px; 
            border-radius: 5px; 
            font-size: 12px;
        }
        .connected { background: #4CAF50; }
        .disconnected { background: #f44336; }
    </style>
</head>
<body>
    <div class="connection-status" id="connectionStatus">Connecting...</div>
    
    <div class="container">
        <div class="header">
            <h1>🤖 QIKI Control Dashboard</h1>
            <p>Advanced Autonomous Bot Simulation Control Interface</p>
        </div>
        
        <div class="status-grid">
            <div class="status-card">
                <div class="status">Position (m)</div>
                <div class="value" id="position">0, 0, 0</div>
            </div>
            <div class="status-card">
                <div class="status">Velocity (m/s)</div>
                <div class="value" id="velocity">0, 0, 0</div>
            </div>
            <div class="status-card">
                <div class="status">Battery (%)</div>
                <div class="value" id="battery">100</div>
            </div>
            <div class="status-card">
                <div class="status">Temperature (°C)</div>
                <div class="value" id="temperature">20</div>
            </div>
            <div class="status-card">
                <div class="status">Integrity (%)</div>
                <div class="value" id="integrity">100</div>
            </div>
            <div class="status-card">
                <div class="status">AI Mode</div>
                <div class="value" id="aiMode">Autonomous</div>
            </div>
        </div>
        
        <div class="control-panel">
            <h3>🎮 Control Panel</h3>
            
            <div class="input-group">
                <label>Set Target Position:</label>
                <input type="text" id="targetInput" placeholder="10, 10, 10" value="10, 10, 10">
                <button class="btn" onclick="setTarget()">Set Target</button>
            </div>
            
            <div>
                <button class="btn" onclick="pauseSimulation()">⏸️ Pause</button>
                <button class="btn" onclick="resumeSimulation()">▶️ Resume</button>
                <button class="btn danger" onclick="emergencyStop()">🛑 Emergency Stop</button>
            </div>
            
            <div style="margin-top: 15px;">
                <label>AI Mode:</label>
                <select id="aiModeSelect" onchange="setAIMode()">
                    <option value="autonomous">Autonomous</option>
                    <option value="supervised">Supervised</option>
                    <option value="manual">Manual</option>
                </select>
            </div>
        </div>
        
        <div id="plot"></div>
    </div>

    <script>
        const socket = io();
        let positionData = { x: [], y: [], z: [], time: [] };
        
        // Подключение к серверу
        socket.on('connect', function() {
            document.getElementById('connectionStatus').textContent = 'Connected';
            document.getElementById('connectionStatus').className = 'connection-status connected';
        });
        
        socket.on('disconnect', function() {
            document.getElementById('connectionStatus').textContent = 'Disconnected';
            document.getElementById('connectionStatus').className = 'connection-status disconnected';
        });
        
        // Обновление данных
        socket.on('update', function(data) {
            updateDisplay(data);
            updatePlot(data);
        });
        
        socket.on('status', function(data) {
            updateDisplay(data);
        });
        
        function updateDisplay(data) {
            document.getElementById('position').textContent = 
                `${data.position[0].toFixed(1)}, ${data.position[1].toFixed(1)}, ${data.position[2].toFixed(1)}`;
            document.getElementById('velocity').textContent = 
                `${data.velocity[0].toFixed(1)}, ${data.velocity[1].toFixed(1)}, ${data.velocity[2].toFixed(1)}`;
            document.getElementById('battery').textContent = data.battery.toFixed(1);
            document.getElementById('temperature').textContent = data.temperature.toFixed(1);
            document.getElementById('integrity').textContent = data.integrity.toFixed(1);
            document.getElementById('aiMode').textContent = data.ai_mode;
        }
        
        function updatePlot(data) {
            const time = Date.now();
            positionData.x.push(data.position[0]);
            positionData.y.push(data.position[1]);
            positionData.z.push(data.position[2]);
            positionData.time.push(time);
            
            // Ограничиваем историю
            if (positionData.x.length > 100) {
                positionData.x.shift();
                positionData.y.shift();
                positionData.z.shift();
                positionData.time.shift();
            }
            
            const traces = [
                { y: positionData.x, name: 'X Position', line: {color: 'red'} },
                { y: positionData.y, name: 'Y Position', line: {color: 'green'} },
                { y: positionData.z, name: 'Z Position', line: {color: 'blue'} }
            ];
            
            const layout = {
                title: 'Position Over Time',
                xaxis: { title: 'Time Steps' },
                yaxis: { title: 'Position (m)' },
                paper_bgcolor: 'rgba(255,255,255,0.1)',
                plot_bgcolor: 'rgba(255,255,255,0.1)',
                font: { color: 'white' }
            };
            
            Plotly.newPlot('plot', traces, layout);
        }
        
        // Функции управления
        function setTarget() {
            const input = document.getElementById('targetInput').value;
            const coords = input.split(',').map(x => parseFloat(x.trim()));
            if (coords.length === 3) {
                fetch('/api/control', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        command: 'set_target',
                        parameters: { target: coords }
                    })
                });
            }
        }
        
        function pauseSimulation() {
            fetch('/api/control', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ command: 'pause' })
            });
        }
        
        function resumeSimulation() {
            fetch('/api/control', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ command: 'resume' })
            });
        }
        
        function emergencyStop() {
            fetch('/api/control', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ command: 'emergency_stop' })
            });
        }
        
        function setAIMode() {
            const mode = document.getElementById('aiModeSelect').value;
            fetch('/api/control', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    command: 'ai_mode',
                    parameters: { mode: mode }
                })
            });
        }
    </script>
</body>
</html>'''
            
            with open(template_path, 'w', encoding='utf-8') as f:
                f.write(html_content)


# Функция для интеграции с существующей симуляцией
def create_web_interface(simulation=None, host='127.0.0.1', port=8080):
    """Создание и запуск веб-интерфейса"""
    return QIKIWebInterface(simulation, host, port)


if __name__ == "__main__":
    # Тестовый запуск
    web_interface = create_web_interface()
    web_interface.start_server()
