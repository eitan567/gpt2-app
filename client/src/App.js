import './App.css';
import React, { useState } from 'react';

function App() {
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState('gpt2');
  const [language, setLanguage] = useState('he');
  const [settings, setSettings] = useState({
    maxLength: 32768,
    temperature: 0.3,
  });
  const [apiKeys, setApiKeys] = useState({
    gemini: '',
  });

  const models = {
    gpt2: { name: 'GPT-2', provider: 'Local' },
    gemini: { name: 'Gemini Pro', provider: 'Google' },
    ollama: {
      name: 'Ollama',
      models: ['llama2', 'mistral', 'codellama', 'phi']
    }
  };

  const handleGeminiGeneration = async (prompt, history) => {
    if (!apiKeys.gemini) {
      throw new Error('Please enter your Gemini API key in the settings');
    }

    const response = await fetch('https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKeys.gemini}`
      },
      body: JSON.stringify({
        contents: [
          ...history.map(msg => ({
            role: msg.type === 'user' ? 'user' : 'model',
            parts: [{ text: msg.content }]
          })),
          {
            role: 'user',
            parts: [{ text: prompt }]
          }
        ],
        generationConfig: {
          temperature: settings.temperature,
          maxOutputTokens: settings.maxLength,
        }
      })
    });

    const data = await response.json();
    if (data.error) {
      throw new Error(data.error.message);
    }
    return data.candidates[0].content.parts[0].text;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!inputText.trim()) return;

    const userMessage = {
      type: 'user',
      content: inputText,
      timestamp: new Date().toLocaleTimeString()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setIsLoading(true);

    try {
      let generatedText;

      if (selectedModel === 'gpt2') {

        const conversationId = localStorage.getItem('conversationId') || crypto.randomUUID();
        localStorage.setItem('conversationId', conversationId);

        // וידוא שהערכים תואמים למגבלות שהגדרנו בשרת
        const requestBody = {
            input_text: inputText,
            max_length: Math.min(Math.max(settings.maxLength, 10), 32768), // בין 10 ל-2000
            temperature: Math.min(Math.max(settings.temperature, 0.1), 1.0), // בין 0.1 ל-1.0
            language: settings.language || "he", // ברירת מחדל "he"
            min_length: 10,  // ערך ברירת מחדל
            length_penalty: 1.0  // ערך ברירת מחדל
        };

        const response = await fetch('http://localhost:8000/generate/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'conversation-id': conversationId
            },
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to generate text');
        }

        const data = await response.json();
        generatedText = data.generated_text;
      } else if (selectedModel.startsWith('gemini')) {
        generatedText = await handleGeminiGeneration(inputText, messages);
      } else if (selectedModel.startsWith('ollama')) {
        const modelName = selectedModel.split('-')[1];
        const response = await fetch('http://localhost:11434/api/generate', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            model: modelName,
            prompt: inputText,
            context: messages.map(m => m.content).join('\n'),
            options: {
              temperature: settings.temperature,
            }
          }),
        });
        const data = await response.json();
        generatedText = data.response;
      }
      
      const aiMessage = {
        type: 'ai',
        content: generatedText,
        timestamp: new Date().toLocaleTimeString()
      };

      setMessages(prev => [...prev, aiMessage]);
    } catch (error) {
      const errorMessage = {
        type: 'error',
        content: 'מצטערים, משהו השתבש. אנא נסו שוב.',
        timestamp: new Date().toLocaleTimeString()
      };
      console.error('Error:', error);
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      <div className="settings-sidebar">
        <div className="settings-section">
          <h2>הגדרות מודל</h2>
          
          <div className="settings-group">
            <h3>בחירת שפה</h3>
            <select 
              value={language} 
              onChange={(e) => setLanguage(e.target.value)}
              className="select-input"
            >
              <option value="he">עברית</option>
              <option value="en">English</option>
            </select>
          </div>

          <div className="settings-group">
            <h3>בחירת מודל</h3>
            <select 
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="select-input"
            >
              <option value="gpt2">GPT-2 (Local)</option>
              <option value="gemini">Gemini Pro</option>
              {models.ollama.models.map(model => (
                <option key={model} value={`ollama-${model}`}>
                  Ollama - {model}
                </option>
              ))}
            </select>
          </div>

          <div className="settings-group">
            <h3>API Keys</h3>
            {selectedModel === 'gemini' && (
              <div className="setting-item">
                <label>Gemini API Key:</label>
                <input
                  type="password"
                  value={apiKeys.gemini}
                  onChange={(e) => setApiKeys(prev => ({
                    ...prev,
                    gemini: e.target.value
                  }))}
                  placeholder="Enter your Gemini API key"
                  className="text-input"
                />
              </div>
            )}
          </div>

          <div className="settings-group">
            <h3>הגדרות מתקדמות</h3>
            <div className="setting-item">
              <label>Temperature:</label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={settings.temperature}
                onChange={(e) => setSettings(prev => ({
                  ...prev,
                  temperature: parseFloat(e.target.value)
                }))}
              />
              <span>{settings.temperature}</span>
            </div>

            <div className="setting-item">
              <label>Max Length:</label>
              <input
                type="number"
                min="100"
                max="32768"
                value={settings.maxLength}
                onChange={(e) => setSettings(prev => ({
                  ...prev,
                  maxLength: parseInt(e.target.value)
                }))}
              />
            </div>
          </div>

          <div className="settings-group">
            <h3>הוראות ראשוניות</h3>
            <div className="instructions">
              <p>1. בחר את המודל הרצוי</p>
              <p>2. הגדר את השפה המועדפת</p>
              <p>3. הזן API Key אם נדרש</p>
              <p>4. התאם את ההגדרות המתקדמות לפי הצורך</p>
              <p>5. התחל בשיחה עם המודל</p>
            </div>
          </div>
        </div>
      </div>

      <div className="chat-container">
        <div className="chat-header">
          <h1>Chat with AI</h1>
        </div>
        
        <div className="messages-container">
          {messages.map((message, index) => (
            <div 
              key={index} 
              className={`message-bubble ${message.type}-message`}
            >
              <div className="message-content">{message.content}</div>
              <div className="message-timestamp">{message.timestamp}</div>
            </div>
          ))}
          {isLoading && (
            <div className="message-bubble ai-message loading">
              <div className="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          )}
        </div>

        <form className="input-form" onSubmit={handleSubmit}>
          <input
            type="text"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="הקלד הודעה כאן..."
            className="message-input"
            dir="rtl"
          />
          <button 
            type="submit" 
            className="send-button"
            disabled={isLoading}
          >
            שלח
          </button>
        </form>
      </div>
    </div>
  );
}

export default App;
