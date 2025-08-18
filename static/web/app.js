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

const SID_KEY = 'chat_sid';
function ensureSid() {
    // URL ?session_id=... 가 있으면 우선 사용, 없으면 탭 단위 세션 보관
    const qs = new URLSearchParams(location.search);
    let id = qs.get('session_id') || sessionStorage.getItem(SID_KEY);
    if (!id) {
        id = (crypto?.randomUUID?.() || Math.random().toString(36).slice(2));
        sessionStorage.setItem(SID_KEY, id);
    }
    return id;
}

async function sendOnce() {
    const q = inputEl.value.trim();
    if (!q) return;
    inputEl.value = '';
    btnEl.disabled = true;

    appendMsg('user', q);
    const bubble = appendMsg('assistant', '');

    // 세션 ID 확보
    let sid = localStorage.getItem('sessionId');
    if (!sid) {
        sid = crypto?.randomUUID?.() || Math.random().toString(36).slice(2);
        localStorage.setItem('sessionId', sid);
    }

    const url = `/api/chat/stream?session_id=${encodeURIComponent(sid)}&message=${encodeURIComponent(q)}`;
    const es = new EventSource(url);

    let endedNormally = false;

    // 서버가 보내는 named event 처리
    es.addEventListener('citations', (e) => {
        // 필요 시: const cits = JSON.parse(e.data);
        // citations UI가 있으면 여기서 반영
    });

    es.addEventListener('token', (e) => {
        bubble.textContent += e.data;         // 한 토큰씩 붙임
        chatEl.scrollTop = chatEl.scrollHeight;
    });

    es.addEventListener('done', () => {
        endedNormally = true;
        es.close();
        btnEl.disabled = false;
    });

    // 백엔드가 기본 message로 보내는 경우(호환)
    es.onmessage = (e) => {
        if (e.data === '[DONE]') {
        endedNormally = true;
        es.close();
        btnEl.disabled = false;
        return;
        }
        try {
        const p = JSON.parse(e.data);
        const chunk = p.delta ?? p.content ?? p.data ?? '';
        bubble.textContent += chunk;
        chatEl.scrollTop = chatEl.scrollHeight;
        } catch {
        bubble.textContent += e.data;
        }
    };

    es.onerror = () => {
        es.close();
        if (!endedNormally) bubble.textContent = '에러: SSE 연결 실패';
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

// RAG 템플릿/번역/Unknown 표기를 걷어내고 정답만 남기는 함수
function sanitizeRagOutput(raw) {
    let s = raw;

    // "Context: ... (중략) ... Answer:" 사이 구간 통째로 제거
    s = s.replace(/Context:[\s\S]*?(?=Answer:|답변:|$)/i, '');

    // "Answer:" / "답변:" 라벨 제거
    s = s.replace(/(?:Answer:|답변:)\s*/gi, '');

    // "번역:" 이후 전부 제거
    s = s.replace(/번역:[\s\S]*$/gi, '');

    // "[n] Unknown" 같은 줄 제거
    s = s.replace(/^\s*\[\d+\]\s*Unknown\s*$/gim, '');

    // 남아있는 중복 빈 줄 정리
    s = s.replace(/\n{3,}/g, '\n\n');

    return s.trim();
}

// sendOnce() 내부에서 bubble 만들고 나서…
let raw = '';

// named event (백엔드: event: token)
es.addEventListener('token', (e) => {
    raw += e.data;
    bubble.textContent = sanitizeRagOutput(raw);
    chatEl.scrollTop = chatEl.scrollHeight;
});

// 호환: 기본 message 이벤트도 지원
es.onmessage = (e) => {
    if (e.data === '[DONE]') {
        es.close();
        btnEl.disabled = false;
        return;
    }
    raw += e.data;
    bubble.textContent = sanitizeRagOutput(raw);
    chatEl.scrollTop = chatEl.scrollHeight;
};
