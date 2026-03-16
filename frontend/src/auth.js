const ALLOWED_DOMAINS = ['talentin.ai', 'avistatech.net', 'loxo.co'];
// SHA-256 of the shared password — plaintext never stored in source
const PASSWORD_HASH = '9cea0340578e9e803e4f9caaf83e51619bba88017c1913d4ce770bfc2dedff54';
const SESSION_KEY = 'jpsearch_user';
const CLIENT_ID_KEY = 'jpsearch_client_id';
const RATE_KEY = 'jpsearch_rate';
const MAX_ATTEMPTS = 5;
const LOCKOUT_MS = 2 * 60 * 1000; // 2 minutes

// ---------------------------------------------------------------------------
// Client isolation: map email address / domain → backend client_id
//
// Priority:  exact email match  >  domain match  >  DEFAULT_CLIENT_ID
//
// Configure via VITE_CLIENT_MAP in your .env file as a JSON string, e.g.:
//   VITE_CLIENT_MAP='{"@talentin.ai":"00","specific@loxo.co":"01"}'
// If VITE_CLIENT_MAP is not set, all internal users default to "00" (Talentin).
// ---------------------------------------------------------------------------
function buildClientMap() {
  try {
    const raw = import.meta.env.VITE_CLIENT_MAP;
    if (raw) return JSON.parse(raw);
  } catch {
    console.warn('VITE_CLIENT_MAP is not valid JSON — using default client map');
  }
  // Built-in defaults — must stay in sync with pipeline/clients.json
  return {
    '@talentin.ai':    '00',
    '@avistatech.net': '00',
    '@loxo.co':        '01',
  };
}

const CLIENT_MAP = buildClientMap();
const DEFAULT_CLIENT_ID = import.meta.env.VITE_DEFAULT_CLIENT_ID || '00';

/**
 * Resolve the backend client_id for the given email.
 * 1. Exact email match (e.g. "boss@acme.com" → "acme")
 * 2. Domain match    (e.g. "@acme.com"      → "acme")
 * 3. Default fallback
 */
export function resolveClientId(email) {
  if (!email) return DEFAULT_CLIENT_ID;
  const lower = email.toLowerCase();
  // 1. Exact email
  if (CLIENT_MAP[lower]) return CLIENT_MAP[lower];
  // 2. Domain
  const domain = '@' + lower.split('@')[1];
  if (CLIENT_MAP[domain]) return CLIENT_MAP[domain];
  // 3. Default
  return DEFAULT_CLIENT_ID;
}

export function getClientId() {
  try {
    // Always re-derive from the stored email so a stale client_id
    // from a previous session / old build can never leak cross-tenant data.
    const email = sessionStorage.getItem(SESSION_KEY);
    if (email) {
      const derived = resolveClientId(email);
      // Self-heal: if stored value is wrong, fix it silently
      const stored = sessionStorage.getItem(CLIENT_ID_KEY);
      if (stored !== derived) {
        sessionStorage.setItem(CLIENT_ID_KEY, derived);
      }
      return derived;
    }
    return sessionStorage.getItem(CLIENT_ID_KEY) || DEFAULT_CLIENT_ID;
  } catch { return DEFAULT_CLIENT_ID; }
}

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
    return { success: false, error: 'Email domain not authorized. Please use your company email.' };
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
  const clientId = resolveClientId(emailLower);
  safeSessionSet(CLIENT_ID_KEY, clientId);
  return { success: true, email: emailLower, clientId };
}

export function getLoggedInUser() {
  return safeSessionGet(SESSION_KEY);
}

export function logout() {
  safeSessionRemove(SESSION_KEY);
  safeSessionRemove(CLIENT_ID_KEY);
}
