import React, { useState } from 'react';

export default function App() {
  const [prompt, setPrompt] = useState('');
  const [feedback, setFeedback] = useState('');
  const [loading, setLoading] = useState(false);
  
  // Workflow states
  const [threadId, setThreadId] = useState('');
  const [currentDraft, setCurrentDraft] = useState(null);
  const [statusMessage, setStatusMessage] = useState('');

  const BACKEND_URL = 'http://localhost:8000';

  // Helper to generate a unique thread ID
  const generateThreadId = () => `thread_${Math.random().toString(36).substr(2, 9)}`;

  // Step 1: Send initial informal notes to start the agent
  const handleStartAgent = async (e) => {
    e.preventDefault();
    if (!prompt.trim() || loading) return;

    setLoading(true);
    setStatusMessage('Generating draft...');
    
    // Fallback to generate a thread ID if none exists yet
    const activeThreadId = threadId || generateThreadId();
    setThreadId(activeThreadId);

    try {
      const response = await fetch(`${BACKEND_URL}/agent/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          thread_id: activeThreadId,
          prompt: prompt,
        }),
      });

      if (!response.ok) throw new Error('Backend error');
      const data = await response.json();
      
      // Fallback text check to support any way your backend returns the string
      const draftText = data.draft || data.content || data.response || JSON.stringify(data, null, 2);
      
      setCurrentDraft(draftText);
      setStatusMessage('Draft generated! Review below or request changes.');
    } catch (error) {
      console.error(error);
      setStatusMessage('Error contacting the AI backend server.');
    } finally {
      setLoading(false);
    }
  };

  // Step 2: Send feedback to resume the agent
  // We pass either true (Approved) or false (Needs Rewrite)
  const handleResumeAgent = async (shouldApprove) => {
    if (loading) return;
    setLoading(true);
    setStatusMessage(shouldApprove ? 'Sending final email...' : 'Updating draft...');

    // If approved, we tell the backend "APPROVED". Otherwise, we pass the text feedback string.
    const feedbackPayload = shouldApprove ? "APPROVED" : feedback;

    try {
      const response = await fetch(`${BACKEND_URL}/agent/resume`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          thread_id: threadId,
          feedback: feedbackPayload,
        }),
      });

      if (!response.ok) throw new Error('Backend error');
      const data = await response.json();
      
      if (shouldApprove) {
        setStatusMessage('🚀 Email sent successfully!');
        setCurrentDraft(null);
        setPrompt('');
        setFeedback('');
        setThreadId(''); // Reset tracking for next fresh email
      } else {
        const updatedDraft = data.draft || data.content || data.response || JSON.stringify(data, null, 2);
        setCurrentDraft(updatedDraft);
        setStatusMessage('Draft updated based on your feedback.');
        setFeedback(''); // Clear the input field for next revision round
      }
    } catch (error) {
      console.error(error);
      setStatusMessage('Error updating the draft.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{
      backgroundColor: '#0f172a',
      color: '#f8fafc',
      minHeight: '100vh',
      fontFamily: 'system-ui, sans-serif',
      padding: '40px 20px',
      display: 'flex',
      justifyContent: 'center'
    }}>
      <div style={{ width: '100%', maxWidth: '650px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
        
        {/* Title */}
        <div style={{ textAlign: 'center' }}>
          <h1 style={{ color: '#818cf8', fontSize: '2rem', margin: '0 0 8px 0' }}>EmailAgent AI Assistant</h1>
          <p style={{ color: '#94a3b8', fontSize: '0.9rem', margin: 0 }}>
            Convert informal notes into professional emails with human-in-the-loop review.
          </p>
        </div>

        {/* Status Messages */}
        {statusMessage && (
          <div style={{
            backgroundColor: statusMessage.includes('🚀') || statusMessage.includes('successfully') ? 'rgba(16, 185, 129, 0.15)' : 'rgba(129, 140, 248, 0.15)',
            border: `1px solid ${statusMessage.includes('🚀') || statusMessage.includes('successfully') ? '#10b981' : '#6366f1'}`,
            color: statusMessage.includes('🚀') || statusMessage.includes('successfully') ? '#34d399' : '#a5b4fc',
            padding: '12px',
            borderRadius: '8px',
            textAlign: 'center',
            fontSize: '0.85rem'
          }}>
            {statusMessage}
          </div>
        )}

        {/* Main Form container */}
        <div style={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '12px', padding: '24px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
          
          <form onSubmit={handleStartAgent} style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            <label style={{ fontSize: '0.75rem', fontWeight: 'bold', color: '#94a3b8', textTransform: 'uppercase' }}>
              What should this email be about?
            </label>
            
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="e.g., tell sarah i cant make the sync today, ask her to drop the slides in slack"
              rows={4}
              disabled={loading || currentDraft !== null}
              style={{
                backgroundColor: '#0f172a',
                border: '1px solid #475569',
                borderRadius: '8px',
                color: '#f8fafc',
                padding: '12px',
                fontSize: '0.9rem',
                resize: 'none',
                opacity: (loading || currentDraft !== null) ? 0.6 : 1
              }}
            />

            {/* Submit button only displays if there is no draft yet */}
            {!currentDraft && (
              <button
                type="submit"
                disabled={loading || !prompt.trim()}
                style={{
                  backgroundColor: (!prompt.trim() || loading) ? '#334155' : '#4f46e5',
                  color: (!prompt.trim() || loading) ? '#94a3b8' : '#ffffff',
                  cursor: (!prompt.trim() || loading) ? 'not-allowed' : 'pointer',
                  border: 'none',
                  borderRadius: '8px',
                  padding: '12px',
                  fontWeight: 'bold',
                  fontSize: '0.9rem',
                  transition: 'background-color 0.2s'
                }}
              >
                {loading ? 'Processing Agent...' : 'Generate Email Draft'}
              </button>
            )}
          </form>

          {/* Step 2 Section: Displays ONLY when a draft is available */}
          {currentDraft && (
            <div style={{ borderTop: '1px solid #334155', paddingTop: '20px', display: 'flex', flexDirection: 'column', gap: '15px' }}>
              
              <div style={{ display: 'flex', justifyContent: 'between', alignItems: 'center', width: '100%' }}>
                <span style={{ fontSize: '0.75rem', fontWeight: 'bold', color: '#94a3b8', textTransform: 'uppercase', flexGrow: 1 }}>
                  Proposed Agent Output
                </span>
                <span style={{ fontSize: '0.75rem', color: '#64748b', fontFamily: 'monospace' }}>
                  ID: {threadId}
                </span>
              </div>

              {/* Draft Box */}
              <div style={{
                backgroundColor: '#020617',
                border: '1px solid #1e293b',
                borderRadius: '8px',
                padding: '16px',
                fontFamily: 'monospace',
                fontSize: '0.9rem',
                color: '#cbd5e1',
                whiteSpace: 'pre-wrap',
                lineHeight: '1.5'
              }}>
                {currentDraft}
              </div>

              {/* Review/Feedback inputs */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                  <label style={{ fontSize: '0.8rem', color: '#94a3b8' }}>
                    Need modifications? Enter adjustments here:
                  </label>
                  <input
                    type="text"
                    value={feedback}
                    onChange={(e) => setFeedback(e.target.value)}
                    placeholder="e.g., Make it formal, change greeting to Team"
                    disabled={loading}
                    style={{
                      backgroundColor: '#0f172a',
                      border: '1px solid #475569',
                      borderRadius: '8px',
                      color: '#f8fafc',
                      padding: '10px',
                      fontSize: '0.9rem'
                    }}
                  />
                </div>

                {/* Branch Operations Buttons */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                  <button
                    type="button"
                    disabled={loading || !feedback.trim()}
                    onClick={() => handleResumeAgent(false)} // isApproved = false (Requests rewrite)
                    style={{
                      backgroundColor: (loading || !feedback.trim()) ? '#334155' : '#475569',
                      color: (loading || !feedback.trim()) ? '#94a3b8' : '#f8fafc',
                      cursor: (loading || !feedback.trim()) ? 'not-allowed' : 'pointer',
                      border: 'none',
                      borderRadius: '8px',
                      padding: '10px',
                      fontWeight: 'bold',
                      fontSize: '0.85rem'
                    }}
                  >
                    Request Rewrite
                  </button>

                  <button
                    type="button"
                    disabled={loading}
                    onClick={() => handleResumeAgent(true)} // isApproved = true (Approves draft)
                    style={{
                      backgroundColor: loading ? '#334155' : '#059669',
                      color: '#ffffff',
                      cursor: loading ? 'not-allowed' : 'pointer',
                      border: 'none',
                      borderRadius: '8px',
                      padding: '10px',
                      fontWeight: 'bold',
                      fontSize: '0.85rem'
                    }}
                  >
                    Approve & Send Email
                  </button>
                </div>
              </div>

              {/* Reset link */}
              <div style={{ textAlign: 'center', marginTop: '5px' }}>
                <span 
                  onClick={() => { setCurrentDraft(null); setPrompt(''); setFeedback(''); setThreadId(''); setStatusMessage(''); }}
                  style={{ color: '#64748b', fontSize: '0.75rem', textDecoration: 'underline', cursor: 'pointer' }}
                >
                  Cancel and start over
                </span>
              </div>

            </div>
          )}

        </div>
      </div>
    </div>
  );
}