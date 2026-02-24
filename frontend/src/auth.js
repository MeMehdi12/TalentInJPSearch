const ALLOWED_DOMAINS = ['talentin.ai', 'avistatech.net'];
// SHA-256 of the shared password — plaintext never stored in source
const PASSWORD_HASH = '9cea0340578e9e803e4f9caaf83e51619bba88017c1913d4ce770bfc2dedff54';
const SESSION_KEY = 'jpsearch_user';
const RATE_KEY = 'jpsearch_rate';
const MAX_ATTEMPTS = 5;
const LOCKOUT_MS = 2 * 60 * 1000; // 2 minutes

async function sha256(str) {
  const buf = new TextEncoder().encode(str);
  const hashBuf = await crypto.subtle.digest('SHA-256', buf);
  return Array.from(new Uint8Array(hashBuf))
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
}

function safeSessionGet(key) {
  try { return sessionStorage.getItem(key); } catch { return null; }
}

function safeSessionSet(key, value) {
  try { sessionStorage.setItem(key, value); } catch { /* storage unavailable */ }
}

function safeSessionRemove(key) {
  try { sessionStorage.removeItem(key); } catch { /* storage unavailable */ }
}

function getRateState() {
  try {
    const raw = safeSessionGet(RATE_KEY);
    return raw ? JSON.parse(raw) : { attempts: 0, lockedUntil: 0 };
  } catch {
    return { attempts: 0, lockedUntil: 0 };
  }
}

function setRateState(state) {
  safeSessionSet(RATE_KEY, JSON.stringify(state));
}

export function isAllowedEmail(email) {
  if (!email || !email.includes('@')) return false;
  const domain = email.split('@')[1]?.toLowerCase();
  return ALLOWED_DOMAINS.includes(domain);
}

export function getLockoutRemaining() {
  const { lockedUntil } = getRateState();
  const remaining = lockedUntil - Date.now();
  return remaining > 0 ? Math.ceil(remaining / 1000) : 0;
}

export async function attemptLogin(email, password) {
  const emailLower = email.trim().toLowerCase();

  // Rate limit check
  const rate = getRateState();
  const lockedSecs = getLockoutRemaining();
  if (lockedSecs > 0) {
    return { success: false, error: `Too many failed attempts. Try again in ${lockedSecs}s.`, locked: true };
  }

  if (!isAllowedEmail(emailLower)) {
    return { success: false, error: 'Email domain not authorized. Only @talentin.ai and @avistatech.net are allowed.' };
  }

  const inputHash = await sha256(password);
  if (inputHash !== PASSWORD_HASH) {
    const newAttempts = rate.attempts + 1;
    const lockedUntil = newAttempts >= MAX_ATTEMPTS ? Date.now() + LOCKOUT_MS : rate.lockedUntil;
    setRateState({ attempts: newAttempts, lockedUntil });
    const remaining = MAX_ATTEMPTS - newAttempts;
    if (remaining <= 0) {
      return { success: false, error: `Too many failed attempts. Account locked for 2 minutes.`, locked: true };
    }
    return { success: false, error: `Incorrect password. ${remaining} attempt${remaining === 1 ? '' : 's'} remaining.` };
  }

  // Success — clear rate state and persist session
  setRateState({ attempts: 0, lockedUntil: 0 });
  safeSessionSet(SESSION_KEY, emailLower);
  return { success: true, email: emailLower };
}

export function getLoggedInUser() {
  return safeSessionGet(SESSION_KEY);
}

export function logout() {
  safeSessionRemove(SESSION_KEY);
}
