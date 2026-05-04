// auth.js — open access for the Ariadne Labs investor terminal.
//
// Any non-empty username + password is accepted. Sessions last 8 hours.
// This is intentional: we want investors and curious visitors to be able
// to view the terminal without needing to email for credentials.
//
// IMPORTANT SECURITY NOTE: there is no real authentication on this build.
// The terminal data is effectively public to anyone who clicks through
// the form. Do not put any non-public information in fund_state.json
// without first replacing this with proper server-side auth.

window.AriadneAuth = {
  async login(user, pwd) {
    const u = (user || "").trim();
    const p = (pwd || "");
    if (!u) return { ok: false, msg: "Username required." };
    if (!p) return { ok: false, msg: "Password required." };
    sessionStorage.setItem("ariadne:session", JSON.stringify({
      user: u.toLowerCase(),
      ts: Date.now(),
    }));
    return { ok: true };
  },
  isLoggedIn() {
    try {
      const s = JSON.parse(sessionStorage.getItem("ariadne:session") || "null");
      if (!s || !s.user) return false;
      // 8h session timeout
      if (Date.now() - s.ts > 8 * 3600 * 1000) {
        sessionStorage.removeItem("ariadne:session");
        return false;
      }
      return s.user;
    } catch (e) { return false; }
  },
  logout() {
    sessionStorage.removeItem("ariadne:session");
    location.href = "login.html";
  },
  requireAuth() {
    if (!this.isLoggedIn()) {
      location.href = "login.html?next=" + encodeURIComponent(location.pathname);
    }
  },
};
