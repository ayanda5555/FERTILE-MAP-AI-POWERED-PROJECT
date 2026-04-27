// ==================== AUTHENTICATION ====================

// Check if user is logged in and update nav
function checkAuth() {
    const token = localStorage.getItem('token');
    const user = JSON.parse(localStorage.getItem('user') || 'null');

    const authNav = document.getElementById('authNav');
    const userNav = document.getElementById('userNav');

    if (authNav && userNav) {
        if (token && user) {
            authNav.style.display = 'none';
            userNav.style.display = 'flex';
        } else {
            authNav.style.display = 'flex';
            userNav.style.display = 'none';
        }
    }
}

// Handle Login
async function handleLogin(event) {
    event.preventDefault();

    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;
    const errorDiv = document.getElementById('loginError');
    const loginBtn = document.getElementById('loginBtn');

    loginBtn.textContent = 'Logging in...';
    loginBtn.disabled = true;
    errorDiv.style.display = 'none';

    try {
        const data = await API.post('/auth/login', { email, password });

        // Save token and user info
        localStorage.setItem('token', data.token);
        localStorage.setItem('user', JSON.stringify(data.user));

        // Redirect to dashboard
        window.location.href = '/pages/dashboard.html';

    } catch (error) {
        errorDiv.textContent = error.message;
        errorDiv.style.display = 'block';
    } finally {
        loginBtn.textContent = 'Login';
        loginBtn.disabled = false;
    }
}

// Handle Registration
async function handleRegister(event) {
    event.preventDefault();

    const fullName = document.getElementById('fullName').value;
    const email = document.getElementById('email').value;
    const farmName = document.getElementById('farmName').value;
    const password = document.getElementById('password').value;
    const confirmPassword = document.getElementById('confirmPassword').value;
    const errorDiv = document.getElementById('registerError');
    const successDiv = document.getElementById('registerSuccess');
    const registerBtn = document.getElementById('registerBtn');

    errorDiv.style.display = 'none';
    successDiv.style.display = 'none';

    // Validate passwords match
    if (password !== confirmPassword) {
        errorDiv.textContent = 'Passwords do not match';
        errorDiv.style.display = 'block';
        return;
    }

    registerBtn.textContent = 'Creating Account...';
    registerBtn.disabled = true;

    try {
        await API.post('/auth/register', {
            full_name: fullName,
            email,
            password,
            farm_name: farmName
        });

        successDiv.textContent = 'Account created successfully! Redirecting to login...';
        successDiv.style.display = 'block';

        setTimeout(() => {
            window.location.href = '/pages/login.html';
        }, 2000);

    } catch (error) {
        errorDiv.textContent = error.message;
        errorDiv.style.display = 'block';
    } finally {
        registerBtn.textContent = 'Create Account';
        registerBtn.disabled = false;
    }
}

// Logout
function logout() {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    window.location.href = '/';
}

// Protect page — redirect if not logged in
function requireAuth() {
    if (!localStorage.getItem('token')) {
        window.location.href = '/pages/login.html';
        return false;
    }
    return true;
}

// Run on page load
document.addEventListener('DOMContentLoaded', checkAuth);