/**
 * Simple tab switching functionality
 * Usage:
 * 1. Import tabs.css and tabs.js
 * 2. Create HTML structure:
 *    - Container with class "tabs-container"
 *    - Unordered list with class "tabs", each list item has a data-tab attribute pointing to the ID of corresponding content
 *    - Content area with class "tab-content", ID matches the data-tab attribute of the corresponding tab
 * 3. Initialize: document.addEventListener('DOMContentLoaded', initTabs);
 */

function initTabs() {
    const tabs = document.querySelectorAll('.tabs li');
    
    // For mobile responsiveness
    function isMobile() {
        return window.innerWidth < 768;
    }
    
    function updateTabsClass() {
        const tabsList = document.querySelectorAll('.tabs');
        tabsList.forEach(list => {
            if (isMobile()) {
                list.classList.add('mobile');
            } else {
                list.classList.remove('mobile');
            }
        });
    }
    
    updateTabsClass();
    window.addEventListener('resize', updateTabsClass);
    
    // Handle tab clicks
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const tabId = tab.getAttribute('data-tab');
            const tabGroup = tab.closest('.tabs-container');
            
            if (tab.parentElement.classList.contains('subtabs')) {
                // If clicking a subtab
                // Only handle the subtab content within the current main tab
                const currentMainTab = tab.closest('.tab-content');
                
                // Remove active class from sibling subtabs only
                const siblingSubtabs = tab.parentElement.querySelectorAll('li');
                siblingSubtabs.forEach(t => t.classList.remove('active'));
                
                // Remove active class from sibling subcontent only
                const siblingSubcontents = currentMainTab.querySelectorAll('.tab-content');
                siblingSubcontents.forEach(content => content.classList.remove('active'));
                
                // Add active class to clicked subtab
                tab.classList.add('active');
                
                // Add active class to corresponding subcontent
                const subcontent = currentMainTab.querySelector(`#${tabId}`);
                if (subcontent) {
                    subcontent.classList.add('active');
                }
            } else {
                // If clicking a main tab
                // Remove active class from all main tabs
                const mainTabs = tabGroup.querySelectorAll('.tabs:not(.subtabs) li');
                mainTabs.forEach(t => t.classList.remove('active'));
                
                // Remove active class from all main tab content
                const mainContents = tabGroup.querySelectorAll('.tabs-container > .tab-content');
                mainContents.forEach(content => content.classList.remove('active'));
                
                // Add active class to clicked main tab
                tab.classList.add('active');
                
                // Add active class to main tab content
                const mainContent = tabGroup.querySelector(`#${tabId}`);
                if (mainContent) {
                    mainContent.classList.add('active');
                    
                    // If no subtab is active, activate the first one
                    if (!mainContent.querySelector('.subtabs li.active')) {
                        const firstSubtab = mainContent.querySelector('.subtabs li');
                        const firstSubcontent = mainContent.querySelector('.subtabs + .tab-content');
                        if (firstSubtab && firstSubcontent) {
                            firstSubtab.classList.add('active');
                            firstSubcontent.classList.add('active');
                        }
                    }
                }
            }
        });
    });
    
    // Initialize first tab and subtab if none are active
    document.querySelectorAll('.tabs-container').forEach(container => {
        const mainTab = container.querySelector('.tabs:not(.subtabs) li');
        if (mainTab && !container.querySelector('.tabs:not(.subtabs) li.active')) {
            mainTab.click();
        }
    });
}

// Initialize tabs when DOM is fully loaded
document.addEventListener('DOMContentLoaded', initTabs);
