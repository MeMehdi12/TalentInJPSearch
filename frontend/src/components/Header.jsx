import React from 'react';

const Header = ({ title, subtitle }) => {
    return (
        <header className="main-header">
            <div className="header-content">
                <h1 className="page-title">{title}</h1>
                {subtitle && <p className="page-subtitle">{subtitle}</p>}
            </div>
            <div className="header-actions">
                <div className="user-badge">DA</div>
            </div>
        </header>
    );
};

export default Header;
