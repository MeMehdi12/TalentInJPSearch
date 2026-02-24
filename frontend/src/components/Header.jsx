import React from 'react';

const Header = ({ title, subtitle, user }) => {
    const initials = user ? user.split('@')[0].slice(0, 2).toUpperCase() : 'JP';
    return (
        <header className="main-header">
            <div className="header-content">
                <h1 className="page-title">{title}</h1>
                {subtitle && <p className="page-subtitle">{subtitle}</p>}
            </div>
            <div className="header-actions">
                <div className="user-badge" title={user}>{initials}</div>
            </div>
        </header>
    );
};

export default Header;
