import React, { useState } from 'react';
import Sidebar from './Sidebar';

const Layout = ({ children, activePage, onNavigate, user }) => {
    const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

    return (
        <div className={`app-layout ${sidebarCollapsed ? 'sidebar-collapsed' : ''}`}>
            <Sidebar
                activePage={activePage}
                onNavigate={onNavigate}
                collapsed={sidebarCollapsed}
                onToggle={() => setSidebarCollapsed(!sidebarCollapsed)}
                user={user}
            />
            <main className="main-content">
                {children}
            </main>
        </div>
    );
};

export default Layout;
