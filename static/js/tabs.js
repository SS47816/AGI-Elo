/**
 * 简单标签切换功能
 * 使用方法：
 * 1. 引入tabs.css和tabs.js
 * 2. 创建HTML结构：
 *    - 包含类名为"tabs-container"的容器
 *    - 包含类名为"tabs"的无序列表，每个列表项有data-tab属性指向对应内容的ID
 *    - 包含类名为"tab-content"的内容区域，ID与对应标签的data-tab属性匹配
 * 3. 初始化：document.addEventListener('DOMContentLoaded', initTabs);
 */

function initTabs() {
    // 获取所有标签和内容
    const tabs = document.querySelectorAll('.tabs li');
    const tabContents = document.querySelectorAll('.tab-content');
    
    // 检测是否为移动设备
    function isMobile() {
        return window.innerWidth < 768;
    }
    
    // 更新标签列表的类
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
    
    // 初始化时检查
    updateTabsClass();
    
    // 监听窗口大小变化
    window.addEventListener('resize', updateTabsClass);
    
    // 为每个标签添加点击事件
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            // 找到当前标签所属的标签组
            const tabGroup = tab.closest('.tabs-container');
            const groupTabs = tabGroup.querySelectorAll('.tabs li');
            const groupContents = tabGroup.querySelectorAll('.tab-content');
            
            // 移除所有标签和内容的活动状态
            groupTabs.forEach(t => t.classList.remove('active'));
            groupContents.forEach(content => content.classList.remove('active'));
            
            // 为当前点击的标签和对应内容添加活动状态
            tab.classList.add('active');
            const tabId = tab.getAttribute('data-tab');
            const content = tabGroup.querySelector(`#${tabId}`);
            if (content) {
                content.classList.add('active');
            }
        });
    });
    
    // 确保至少有一个标签处于活动状态
    document.querySelectorAll('.tabs-container').forEach(container => {
        const containerTabs = container.querySelectorAll('.tabs li');
        const containerContents = container.querySelectorAll('.tab-content');
        
        if (containerTabs.length > 0 && !container.querySelector('.tabs li.active')) {
            containerTabs[0].classList.add('active');
            if (containerContents.length > 0) {
                containerContents[0].classList.add('active');
            }
        }
    });
}

// 当DOM加载完成后初始化标签
document.addEventListener('DOMContentLoaded', initTabs);
