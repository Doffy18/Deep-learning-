import { useState } from "react";

type Message = {
  role: "user" | "assistant";
  content: string;
};

const API_BASE_URL = "http://localhost:8000";

function App() {
  const [videoUrl, setVideoUrl] = useState("");
  const [lastProcessedUrl, setLastProcessedUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [question, setQuestion] = useState("");
  const [videoInfo, setVideoInfo] = useState({
    title: "",
    thumbnail: "",
  });

  const processVideo = async () => {
    if (!videoUrl) return;
    setLoading(true);

    if (videoUrl.trim() !== lastProcessedUrl.trim()) {
      setMessages([]);
    }

    try {
      const response = await fetch(`${API_BASE_URL}/transcript`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ video_url: videoUrl.trim() }),
      });

      if (!response.ok) {
        throw new Error("Failed to process transcript on server side.");
      }

      const data = await response.json();
      
      setVideoInfo({
        title: data.title,
        thumbnail: data.thumbnail,
      });
      setLastProcessedUrl(videoUrl);
      setLoading(false);

    } catch (err) {
      console.error("Backend Error:", err);
      setLoading(false);
    }
  };

  const askQuestion = async () => {
    if (!question.trim()) return;

    const userMessage: Message = {
      role: "user",
      content: question,
    };

    setMessages((prev) => [...prev, userMessage]);
    const currentQuestion = question;
    setQuestion("");

    try {
      const response = await fetch(`${API_BASE_URL}/query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ user_query: currentQuestion }),
      });

      if (!response.ok) {
        throw new Error("Failed to parse query response.");
      }

      const data = await response.json();

      const aiMessage: Message = {
        role: "assistant",
        content: data.answer,
      };
      
      setMessages((prev) => [...prev, aiMessage]);

    } catch (err) {
      console.error("Backend Error:", err);
    }
  };

  const handleSystemRefresh = () => {
    setVideoUrl("");
    setLastProcessedUrl("");
    setMessages([]);
    setVideoInfo({ title: "", thumbnail: "" });
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "linear-gradient(135deg, #0f172a 0%, #020617 100%)",
        color: "#f8fafc",
        padding: "40px 20px",
        fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
      }}
    >
      <div style={{ width: "100%", maxWidth: "800px" }}>
        
        {/* Header Section */}
        <header style={{ marginBottom: "40px", textAlign: "center" }}>
          <h1
            style={{
              fontSize: "2.5rem",
              fontWeight: "800",
              letterSpacing: "-0.05em",
              background: "linear-gradient(to right, #38bdf8, #818cf8)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
              margin: "0 0 10px 0",
            }}
          >
            YouTube Video RAG
          </h1>
          <p style={{ color: "#94a3b8", fontSize: "1rem", margin: 0 }}>
            Extract insights and chat dynamically with any YouTube video content.
          </p>
        </header>

        {/* URL Input Box */}
        <div
          style={{
            display: "flex",
            gap: "12px",
            background: "#1e293b",
            padding: "8px",
            borderRadius: "16px",
            boxShadow: "0 4px 30px rgba(0, 0, 0, 0.3)",
            border: "1px solid #334155",
            marginBottom: "30px",
          }}
        >
          <input
            value={videoUrl}
            onChange={(e) => setVideoUrl(e.target.value)}
            placeholder="Paste YouTube video link here..."
            style={{
              flex: 1,
              padding: "14px 16px",
              borderRadius: "12px",
              background: "transparent",
              border: "none",
              color: "#f8fafc",
              fontSize: "0.95rem",
              outline: "none",
            }}
          />
          <button
            onClick={processVideo}
            disabled={loading}
            style={{
              padding: "0 24px",
              borderRadius: "12px",
              background: "linear-gradient(135deg, #6366f1 0%, #4f46e5 100%)",
              color: "white",
              fontWeight: "600",
              fontSize: "0.95rem",
              border: "none",
              cursor: loading ? "not-allowed" : "pointer",
              transition: "all 0.2s ease",
              opacity: loading ? 0.7 : 1,
              boxShadow: "0 4px 12px rgba(79, 70, 229, 0.3)",
            }}
          >
            {loading ? "Analyzing..." : "Process Video"}
          </button>
        </div>

        {/* Loading Spinner State */}
        {loading && (
          <div style={{ textAlign: "center", padding: "20px 0", color: "#38bdf8" }}>
            <div
              style={{
                width: "24px",
                height: "24px",
                border: "3px solid #334155",
                borderTopColor: "#38bdf8",
                borderRadius: "50%",
                display: "inline-block",
                animation: "spin 1s linear infinite",
                marginRight: "10px",
                verticalAlign: "middle",
              }}
            />
            <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
            <span style={{ fontWeight: "500" }}>Ingesting video data & building vector index...</span>
          </div>
        )}

        {/* Dashboard layout once Video is Ready */}
        {videoInfo.title && (
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr",
              gap: "24px",
              animation: "fadeIn 0.5s ease-out",
            }}
          >
            <style>{`@keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }`}</style>

            {/* Video Overview Card */}
            <div
              style={{
                background: "linear-gradient(145deg, #1e293b 0%, #0f172a 100%)",
                padding: "24px",
                borderRadius: "20px",
                border: "1px solid #334155",
                display: "flex",
                gap: "24px",
                flexWrap: "wrap",
              }}
            >
              <img
                src={videoInfo.thumbnail}
                alt="Video Thumbnail"
                style={{
                  width: "220px",
                  height: "125px",
                  objectFit: "cover",
                  borderRadius: "12px",
                  border: "1px solid #475569",
                  boxShadow: "0 8px 16px rgba(0,0,0,0.4)",
                }}
              />
              <div style={{ flex: 1, minWidth: "250px" }}>
                <span style={{ fontSize: "0.75rem", textTransform: "uppercase", color: "#38bdf8", fontWeight: "700" }}>Active Context</span>
                <h2 style={{ fontSize: "1.25rem", fontWeight: "700", margin: "4px 0 16px 0", lineHeight: "1.4", color: "#f1f5f9" }}>{videoInfo.title}</h2>
                
                {/* Pipeline Badges */}
                <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
                  {["Transcript Extracted", "Chunks Created", "Vector DB Ready"].map((text, i) => (
                    <span
                      key={i}
                      style={{
                        padding: "4px 12px",
                        borderRadius: "20px",
                        background: "rgba(16, 185, 129, 0.1)",
                        border: "1px solid rgba(16, 185, 129, 0.3)",
                        color: "#34d399",
                        fontSize: "0.75rem",
                        fontWeight: "600",
                      }}
                    >
                      ✓ {text}
                    </span>
                  ))}
                </div>
              </div>
            </div>

            {/* Conversational AI Workspace */}
            <div
              style={{
                background: "#1e293b",
                borderRadius: "20px",
                border: "1px solid #334155",
                overflow: "hidden",
                boxShadow: "0 10px 25px -5px rgba(0, 0, 0, 0.3)",
              }}
            >
              {/* Chat Title bar + Interactive Refresh Option */}
              <div 
                style={{ 
                  padding: "16px 24px", 
                  borderBottom: "1px solid #334155", 
                  background: "#151f32",
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center"
                }}
              >
                <h3 style={{ margin: 0, fontSize: "1rem", fontWeight: "600", color: "#94a3b8" }}>Cognitive Engine Chat</h3>
                <button
                  onClick={handleSystemRefresh}
                  style={{
                    background: "transparent",
                    border: "1px solid #475569",
                    borderRadius: "6px",
                    color: "#94a3b8",
                    padding: "4px 12px",
                    fontSize: "0.8rem",
                    fontWeight: "600",
                    cursor: "pointer",
                    transition: "all 0.2s"
                  }}
                  onMouseEnter={(e) => (e.currentTarget.style.borderColor = "#f43f5e", e.currentTarget.style.color = "#f43f5e")}
                  onMouseLeave={(e) => (e.currentTarget.style.borderColor = "#475569", e.currentTarget.style.color = "#94a3b8")}
                >
                  ↻ Refresh Session
                </button>
              </div>

              {/* Messages Container */}
              <div
                style={{
                  height: "350px",
                  overflowY: "auto",
                  padding: "24px",
                  display: "flex",
                  flexDirection: "column",
                  gap: "16px",
                  background: "#0f172a",
                }}
              >
                {messages.length === 0 ? (
                  <div style={{ margin: "auto", textAlign: "center", color: "#475569" }}>
                    <p style={{ fontSize: "0.95rem", margin: 0 }}>No questions asked yet. Query the database below.</p>
                  </div>
                ) : (
                  messages.map((msg, index) => {
                    const isUser = msg.role === "user";
                    return (
                      <div
                        key={index}
                        style={{
                          display: "flex",
                          flexDirection: "column",
                          alignItems: isUser ? "flex-end" : "flex-start",
                        }}
                      >
                        <span
                          style={{
                            fontSize: "0.75rem",
                            color: "#64748b",
                            fontWeight: "600",
                            marginBottom: "4px",
                          }}
                        >
                          {isUser ? "YOU" : "KNOWLEDGE ASSISTANT"}
                        </span>
                        <div
                          style={{
                            maxWidth: "80%",
                            padding: "12px 16px",
                            borderRadius: isUser ? "16px 16px 2px 16px" : "16px 16px 16px 2px",
                            background: isUser ? "#4f46e5" : "#1e293b",
                            border: isUser ? "none" : "1px solid #334155",
                            color: isUser ? "white" : "#e2e8f0",
                            fontSize: "0.95rem",
                            lineHeight: "1.5",
                          }}
                        >
                          {msg.content}
                        </div>
                      </div>
                    );
                  })
                )}
              </div>

              {/* Chat Form Footer */}
              <div
                style={{
                  padding: "16px 24px",
                  background: "#151f32",
                  borderTop: "1px solid #334155",
                  display: "flex",
                  gap: "12px",
                }}
              >
                <input
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder="Ask a question about the video's contents..."
                  onKeyDown={(e) => e.key === "Enter" && askQuestion()}
                  style={{
                    flex: 1,
                    padding: "12px 16px",
                    borderRadius: "10px",
                    background: "#0f172a",
                    border: "1px solid #334155",
                    color: "#f8fafc",
                    fontSize: "0.95rem",
                    outline: "none",
                  }}
                />
                <button
                  onClick={askQuestion}
                  style={{
                    padding: "0 20px",
                    borderRadius: "10px",
                    background: "#38bdf8",
                    color: "#0f172a",
                    fontWeight: "700",
                    fontSize: "0.95rem",
                    border: "none",
                    cursor: "pointer",
                    transition: "all 0.2s",
                  }}
                >
                  Send
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;