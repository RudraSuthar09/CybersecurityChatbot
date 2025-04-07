import React, { useEffect, useRef, useState } from 'react';
import MessageBubble from './MessageBubble';

const ChatbotWidget = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [socketConnected, setSocketConnected] = useState(false);
  const socketRef = useRef(null);

  useEffect(() => {
    const socket = new WebSocket('ws://localhost:4000');
    socketRef.current = socket;

    socket.onopen = () => {
      console.log('🟢 Connected to WebSocket Server');
      setSocketConnected(true);
    };

    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        // ✅ Handle structured bot response
        const { message, class_code, class_name, confidence } = data;

        // Log optional analytics/debug info
        console.log("📊 Threat Class:", class_name);
        console.log("🆔 Class Code:", class_code);
        console.log("🎯 Confidence:", confidence);

        setMessages((prev) => [...prev, { sender: 'bot', text: message }]);
      } catch (err) {
        console.error('⚠️ Error parsing message:', err);
      }
    };

    socket.onerror = (error) => {
      console.error('❌ WebSocket Error:', error);
    };

    socket.onclose = () => {
      console.warn('🔌 WebSocket connection closed');
      setSocketConnected(false);
    };

    return () => {
      socket.close();
    };
  }, []);

  const handleSend = () => {
    if (input.trim() === '') return;

    const userMessage = { sender: 'user', text: input };
    setMessages((prev) => [...prev, userMessage]);

    if (socketConnected && socketRef.current.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify({ message: input }));
    } else {
      console.warn('WebSocket is not open. Message not sent.');
    }

    setInput('');
  };

  return (
    <div className="fixed bottom-4 right-4 w-80 bg-white shadow-xl rounded-2xl p-4 z-50">
      <div className="h-96 overflow-y-auto mb-4 space-y-2">
        {messages.map((msg, index) => (
          <MessageBubble key={index} sender={msg.sender} text={msg.text} />
        ))}
      </div>
      <div className="flex gap-2">
        <input
          type="text"
          className="flex-1 border rounded-lg px-3 py-2"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSend()}
          placeholder={
            socketConnected ? 'Ask about cybersecurity...' : 'Connecting...'
          }
          disabled={!socketConnected}
        />
        <button
          className="bg-blue-600 text-white px-4 py-2 rounded-lg"
          onClick={handleSend}
          disabled={!socketConnected}
        >
          Send
        </button>
      </div>
    </div>
  );
};

export default ChatbotWidget;
