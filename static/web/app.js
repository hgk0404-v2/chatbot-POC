const chatEl = document.getElementById('chat');
const inputEl = document.getElementById('prompt');
const btnEl = document.getElementById('sendBtn');

let sessionId = localStorage.getItem('sessionId');
if (!sessionId) {
    sessionId = Math.random().toString(36).slice(2);
    localStorage.setItem('sessionId', sessionId);
}

function appendMsg(role, text) {
    const row = document.createElement('div');
    row.className = `msg ${role}`;
    const b = document.createElement('div');
    b.className = 'bubble';
    b.textContent = text;
    row.appendChild(b);
    chatEl.appendChild(row);
    chatEl.scrollTop = chatEl.scrollHeight;
    return b; // bubble 엘리먼트 반환(스트리밍시 업데이트)
}

async function sendOnce() {
    const q = inputEl.value.trim();
    if (!q) return;
    inputEl.value = ''; btnEl.disabled = true;

    appendMsg('user', q);
    const bubble = appendMsg('assistant', '');

    // ✅ 절대경로 + GET + SSE
    const url = `/chat/stream?session_id=${encodeURIComponent(sessionId)}&message=${encodeURIComponent(q)}`;
    const es = new EventSource(url);

    es.addEventListener('delta', (e) => {
        bubble.textContent += e.data || '';
        chatEl.scrollTop = chatEl.scrollHeight;
    });
    es.addEventListener('done', () => {
        es.close();
        btnEl.disabled = false;
    });
    es.onerror = () => {
        es.close();
        bubble.textContent = '에러: SSE 연결 실패';
        btnEl.disabled = false;
    };
}


btnEl.addEventListener('click', sendOnce);
inputEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendOnce();
    }
});
