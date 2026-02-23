import React, { useState, useEffect } from 'react';
import { IconMapPin, IconBriefcase, IconLinkedin, IconChevronDown, IconChevronUp, IconCamera, IconImage, IconCheckCircle, IconXCircle } from './Icons';
import { getPersonCertifications, getPersonEducations, getPersonRoles, getPersonSkills } from '../api';

const CandidateCard = ({ person }) => {
    const [isExpanded, setIsExpanded] = useState(false);
    const [additionalData, setAdditionalData] = useState({
        certifications: [],
        educations: [],
        roles: [],
        skills: [],
        loading: false,
        loaded: false
    });
    const [expandedDescriptions, setExpandedDescriptions] = useState({});

    const toggleExpand = (e) => {
        // Prevent toggle if clicking on links
        if (e.target.closest('a')) return;
        setIsExpanded(!isExpanded);

        // Load additional data when expanding for the first time
        if (!isExpanded && !additionalData.loaded && person.forager_id) {
            loadAdditionalData();
        }
    };

    const loadAdditionalData = async () => {
        setAdditionalData(prev => ({ ...prev, loading: true }));
        try {
            const [certs, edus, roles, skills] = await Promise.all([
                getPersonCertifications(person.forager_id),
                getPersonEducations(person.forager_id),
                getPersonRoles(person.forager_id),
                getPersonSkills(person.forager_id)
            ]);

            // Deduplicate roles based on role_title, organization_name, and start_date
            const uniqueRoles = roles ? roles.filter((role, index, self) =>
                index === self.findIndex((r) => (
                    r.role_title === role.role_title &&
                    r.organization_name === role.organization_name &&
                    r.start_date === role.start_date
                ))
            ) : [];

            // Deduplicate skills
            const uniqueSkills = skills ? [...new Set(skills)] : [];

            setAdditionalData({
                certifications: certs || [],
                educations: edus || [],
                roles: uniqueRoles,
                skills: uniqueSkills,
                loading: false,
                loaded: true
            });
        } catch (err) {
            console.error('Failed to load additional data:', err);
            setAdditionalData(prev => ({ ...prev, loading: false, loaded: true }));
        }
    };

    // Derive initials from full_name for accuracy
    const displayName = person.full_name || `${person.first_name || ''} ${person.last_name || ''}`.trim() || 'Unknown';
    const nameParts = displayName.split(/\s+/);
    const initials = nameParts.length >= 2
        ? `${nameParts[0][0] || ''}${nameParts[nameParts.length - 1][0] || ''}`.toUpperCase()
        : (nameParts[0]?.[0] || 'U').toUpperCase();

    // Common acronyms to preserve in uppercase
    const ACRONYMS = new Set(['IT', 'HR', 'AI', 'ML', 'US', 'UK', 'EU', 'MBA', 'CPA', 'CEO', 'CTO', 'CFO', 'COO', 'VP', 'SVP', 'EVP', 'MD', 'PhD', 'BS', 'MS', 'BA', 'MA', 'USCPA', 'AWS', 'GCP', 'API', 'SQL', 'CSS', 'HTML', 'PM', 'QA', 'UX', 'UI']);

    const formatText = (str) => {
        if (!str) return '';
        if (typeof str !== 'string') return String(str);
        str = str.replace(/^["']|["']$/g, '');
        if (str === 'Unknown' || str === 'unknown') return '';

        return str.split(/\s+/).map(word => {
            const upper = word.toUpperCase();
            // Keep known acronyms
            if (ACRONYMS.has(upper)) return upper;
            // Title case everything else
            return word.charAt(0).toUpperCase() + word.slice(1).toLowerCase();
        }).join(' ');
    };

    const parseSkill = (skill) => {
        if (!skill) return '';

        // If it's already an object
        if (typeof skill === 'object') {
            // Handle {skill: "..."} from backend
            const rawValue = skill.skill || skill.skill_name || skill.name || skill.Skill;
            if (rawValue && typeof rawValue === 'string') {
                // The value itself might be JSON like {"Skill":"Microsoft Excel"}
                if (rawValue.trim().startsWith('{')) {
                    try {
                        const inner = JSON.parse(rawValue);
                        return inner.Skill || inner.skill_name || inner.name || Object.values(inner)[0] || rawValue;
                    } catch (e) {
                        return rawValue;
                    }
                }
                return rawValue;
            }
            // Fallback: try any value
            return Object.values(skill).find(v => typeof v === 'string' && v.trim()) || JSON.stringify(skill);
        }

        // If it's a string that looks like JSON
        if (typeof skill === 'string' && skill.trim().startsWith('{')) {
            try {
                const parsed = JSON.parse(skill);
                return parsed.Skill || parsed.skill_name || parsed.name || parsed.skill || Object.values(parsed)[0] || skill;
            } catch (e) {
                return skill;
            }
        }

        return skill;
    };

    const safeDate = (dateStr) => {
        if (!dateStr) return null;
        try {
            const d = new Date(dateStr);
            if (isNaN(d.getTime()) || d.getFullYear() < 1900) return null;
            return d;
        } catch (e) {
            return null;
        }
    };

    const formatDate = (dateStr, format = 'month-year') => {
        const d = safeDate(dateStr);
        if (!d) return 'Present'; // Or 'Unknown' depending on context

        if (format === 'full') {
            return d.toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' });
        }
        return d.toLocaleDateString('en-US', { year: 'numeric', month: 'short' });
    };

    return (
        <div className={`candidate-card ${isExpanded ? 'expanded' : ''}`} onClick={toggleExpand}>
            <div className="card-main">
                <div className="card-header-row">
                    <div className="card-avatar">
                        <img
                            src={person.photo || `https://ui-avatars.com/api/?name=${person.first_name}+${person.last_name}&background=random&color=fff&size=128`}
                            alt={`${person.first_name} ${person.last_name}`}
                            onError={(e) => {
                                e.target.onerror = null;
                                e.target.src = `https://ui-avatars.com/api/?name=${person.first_name}+${person.last_name}&background=random&color=fff&size=128`;
                            }}
                        />
                    </div>

                    <div className="card-info">
                        <div className="card-top-line">
                            <h3 className="card-name">
                                {person.full_name ? formatText(person.full_name) :
                                    (person.first_name || person.last_name) ? `${formatText(person.first_name)} ${formatText(person.last_name)}`.trim() :
                                        formatText(person.search_name) || 'Unknown'}
                            </h3>
                        </div>
                        <p className="card-headline">{formatText(person.headline) || 'No headline available'}</p>

                        <div className="card-meta">
                            {(formatText(person.city) || formatText(person.country)) && (
                                <span className="meta-item">
                                    <IconMapPin size={14} /> {[formatText(person.city), formatText(person.country)].filter(Boolean).join(', ')}
                                </span>
                            )}
                            {person.industry && (
                                <span className="meta-item">
                                    <IconBriefcase size={14} /> {formatText(person.industry)}
                                </span>
                            )}
                        </div>
                    </div>

                    <div className="card-expand-icon">
                        {isExpanded ? <IconChevronUp size={20} /> : <IconChevronDown size={20} />}
                    </div>
                </div>
            </div>

            <div className={`card-details-wrapper ${isExpanded ? 'show' : ''}`}>
                <div className="card-details-content">
                    {person.description && (
                        <div className="detail-section">
                            <h4>About</h4>
                            <p>{person.description}</p>
                        </div>
                    )}

                    <div className="detail-section">
                        <h4>Location Details</h4>
                        <div className="detail-grid">
                            <div className="detail-item">
                                <span className="label">Country</span>
                                <span className="value">{formatText(person.country) || '-'}</span>
                            </div>
                            <div className="detail-item">
                                <span className="label">City</span>
                                <span className="value">{formatText(person.city) || '-'}</span>
                            </div>
                            <div className="detail-item">
                                <span className="label">Area</span>
                                <span className="value">{formatText(person.area) || '-'}</span>
                            </div>
                            <div className="detail-item">
                                <span className="label">Location</span>
                                <span className="value">{formatText(person.location) || '-'}</span>
                            </div>
                            <div className="detail-item">
                                <span className="label">LinkedIn Country</span>
                                <span className="value">{formatText(person.linkedin_country) || '-'}</span>
                            </div>
                            <div className="detail-item">
                                <span className="label">LinkedIn Area</span>
                                <span className="value">{formatText(person.linkedin_area) || '-'}</span>
                            </div>
                            {person.address && (
                                <div className="detail-item">
                                    <span className="label">Address</span>
                                    <span className="value">{formatText(person.address)}</span>
                                </div>
                            )}
                        </div>
                    </div>

                    <div className="detail-section">
                        <h4>Professional Info</h4>
                        <div className="detail-grid">
                            <div className="detail-item">
                                <span className="label">Industry</span>
                                <span className="value">{formatText(person.industry) || '-'}</span>
                            </div>
                            <div className="detail-item">
                                <span className="label">LinkedIn Slug</span>
                                <span className="value">{person.linkedin_slug || '-'}</span>
                            </div>
                            <div className="detail-item">
                                <span className="label">Search Name</span>
                                <span className="value">{formatText(person.search_name) || '-'}</span>
                            </div>
                        </div>
                    </div>

                    <div className="detail-section">
                        <h4>Profile Status</h4>
                        <div className="detail-grid">
                            <div className="detail-item">
                                <span className="label">Creator</span>
                                <span className="value status-value">
                                    {person.is_creator ? <><IconCheckCircle size={16} className="text-success" /> Yes</> : <><IconXCircle size={16} className="text-muted" /> No</>}
                                </span>
                            </div>
                            <div className="detail-item">
                                <span className="label">Influencer</span>
                                <span className="value status-value">
                                    {person.is_influencer ? <><IconCheckCircle size={16} className="text-success" /> Yes</> : <><IconXCircle size={16} className="text-muted" /> No</>}
                                </span>
                            </div>
                            <div className="detail-item">
                                <span className="label">Last Updated</span>
                                <span className="value">{person.date_updated ? formatDate(person.date_updated, 'full') : '-'}</span>
                            </div>
                            {person.primary_locale && (
                                <div className="detail-item">
                                    <span className="label">Locale</span>
                                    <span className="value">{person.primary_locale}</span>
                                </div>
                            )}
                            {person.temporary_status && (
                                <div className="detail-item">
                                    <span className="label">Status</span>
                                    <span className="value">{person.temporary_emoji_status ? `${person.temporary_emoji_status} ${formatText(person.temporary_status)}` : formatText(person.temporary_status)}</span>
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Skills Section */}
                    {additionalData.loading && <div className="detail-section"><p>Loading additional information...</p></div>}
                    {additionalData.loaded && additionalData.skills.length > 0 && (
                        <div className="detail-section">
                            <h4>Skills ({additionalData.skills.length})</h4>
                            <div className="skills-container">
                                {additionalData.skills.map((skill, idx) => (
                                    <span key={idx} className="skill-badge">{formatText(parseSkill(skill))}</span>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Roles Section */}
                    {additionalData.loaded && additionalData.roles.length > 0 && (
                        <div className="detail-section">
                            <h4>Work Experience ({additionalData.roles.length})</h4>
                            <div className="roles-scrollable-container">
                                {additionalData.roles.map((role, idx) => {
                                    return (
                                        <div key={idx} className="experience-item">
                                            <div className="experience-header">
                                                <h5>{formatText(role.role_title || role.title || role.job_title) || 'Position'}</h5>
                                                <p className="date-range">
                                                    {safeDate(role.start_date) ? formatDate(role.start_date) : 'Unknown'} - {role.is_current ? 'Present' : (safeDate(role.end_date) ? formatDate(role.end_date) : 'Present')}
                                                </p>
                                            </div>
                                            <p className="org-name">{formatText(role.organization_name || role.company || role.company_name) || '-'}</p>
                                            {role.description && (
                                                <div>
                                                    <p className="role-description">
                                                        {expandedDescriptions[idx] ? role.description : role.description.substring(0, 200)}
                                                        {role.description.length > 200 && !expandedDescriptions[idx] && '...'}
                                                    </p>
                                                    {role.description.length > 200 && (
                                                        <button
                                                            className="btn-link text-muted mt-1"
                                                            onClick={(e) => {
                                                                e.stopPropagation();
                                                                setExpandedDescriptions(prev => ({ ...prev, [idx]: !prev[idx] }));
                                                            }}
                                                            style={{ background: 'none', border: 'none', padding: 0, cursor: 'pointer', textDecoration: 'underline', fontSize: '0.85rem' }}
                                                        >
                                                            {expandedDescriptions[idx] ? 'Show Less' : 'Show More'}
                                                        </button>
                                                    )}
                                                </div>
                                            )}
                                        </div>
                                    );
                                })}
                            </div>
                        </div>
                    )}

                    {/* Education Section */}
                    {additionalData.loaded && additionalData.educations.length > 0 && (
                        <div className="detail-section">
                            <h4>Education ({additionalData.educations.length})</h4>
                            <div className="roles-scrollable-container">
                                {additionalData.educations.map((edu, idx) => (
                                    <div key={idx} className="education-item">
                                        <h5>{formatText(edu.school_name || edu.school) || 'School'}</h5>
                                        <p className="degree-info">{formatText(edu.degree) || formatText(edu.field_of_study) || '-'}</p>
                                        {edu.field_of_study && edu.degree && <p className="field-info">{formatText(edu.field_of_study)}</p>}
                                        <p className="date-range">
                                            {safeDate(edu.start_date) ? safeDate(edu.start_date).getFullYear() : '-'} - {safeDate(edu.end_date) ? safeDate(edu.end_date).getFullYear() : 'Present'}
                                        </p>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Certifications Section */}
                    {additionalData.loaded && additionalData.certifications.length > 0 && (
                        <div className="detail-section">
                            <h4>Certifications ({additionalData.certifications.length})</h4>
                            <div className="roles-scrollable-container">
                                {additionalData.certifications.map((cert, idx) => (
                                    <div key={idx} className="certification-item">
                                        <h5>{formatText(cert.certificate_name || cert.name) || 'Certificate'}</h5>
                                        <p className="cert-authority">{formatText(cert.authority) || '-'}</p>
                                        {cert.start && <p className="cert-date">Issued: {formatDate(cert.start)}</p>}
                                        {cert.certificate_url && (
                                            <a href={cert.certificate_url} target="_blank" rel="noopener noreferrer" className="cert-link">View Certificate</a>
                                        )}
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {(person.linkedin_url || person.photo || person.background_picture) && (
                        <div className="detail-section">
                            <h4>Links & Media</h4>
                            <div className="detail-links">
                                {person.linkedin_url && (
                                    <a
                                        href={person.linkedin_url}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="btn-linkedin"
                                    >
                                        <IconLinkedin size={16} /> View LinkedIn Profile
                                    </a>
                                )}
                                {person.photo && (
                                    <a href={person.photo} target="_blank" rel="noopener noreferrer" className="btn-link">
                                        <IconCamera size={16} /> Profile Photo
                                    </a>
                                )}
                                {person.background_picture && (
                                    <a href={person.background_picture} target="_blank" rel="noopener noreferrer" className="btn-link">
                                        <IconImage size={16} /> Cover Image
                                    </a>
                                )}
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default CandidateCard;

