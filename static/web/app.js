async function send() {
    const input = document.getElementById('userInput');
    const q = (input?.value || '').trim();
    if (!q) return;

    const res = await fetch('/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ message: q, session_id: 'dev' })
    });

    if (!res.ok) {
        document.getElementById('chat').textContent = `에러: ${res.status}`;
        return;
    }

    const data = await res.json();
    document.getElementById('chat').textContent = data.answer || '(응답 없음)';
}
