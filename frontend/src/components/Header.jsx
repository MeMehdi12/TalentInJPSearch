import React from 'react';

const Header = ({ title, subtitle, user, onLogout }) => {
    const initials = user ? user.split('@')[0].slice(0, 2).toUpperCase() : 'JP';
    return (
        <header className="main-header">
            <div className="header-content">
                <h1 className="page-title">{title}</h1>
                {subtitle && <p className="page-subtitle">{subtitle}</p>}
            </div>
            <div className="header-actions" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                {user && (
                    <span style={{ fontSize: '0.8rem', color: '#6b7280', maxWidth: '180px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {user}
                    </span>
                )}
                <div className="user-badge" title={user}>{initials}</div>
                {onLogout && (
                    <button
                        onClick={onLogout}
                        title="Sign out"
                        style={{
                            background: 'none',
                            border: '1.5px solid #e5e7eb',
                            borderRadius: '0.5rem',
                            padding: '0.35rem 0.75rem',
                            fontSize: '0.78rem',
                            fontWeight: '600',
                            color: '#6b7280',
                            cursor: 'pointer',
                        }}
                    >
                        Sign out
                    </button>
                )}
            </div>
        </header>
    );
};

export default Header;
