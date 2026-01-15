import { ReactNode } from 'react'
import { Link, useLocation } from 'react-router-dom'
import {
    HomeIcon,
    DocumentTextIcon,
    SparklesIcon,
    QuestionMarkCircleIcon,
    Bars3Icon,
    XMarkIcon
} from '@heroicons/react/24/outline'
import { useAppStore } from '../store'
import clsx from 'clsx'

interface LayoutProps {
    children: ReactNode
}

const navigation = [
    { name: 'Dashboard', href: '/', icon: HomeIcon },
    { name: 'Tài liệu', href: '/documents', icon: DocumentTextIcon },
    { name: 'Tạo câu hỏi', href: '/generate', icon: SparklesIcon },
    { name: 'Ngân hàng câu hỏi', href: '/questions', icon: QuestionMarkCircleIcon },
]

export default function Layout({ children }: LayoutProps) {
    const location = useLocation()
    const { sidebarOpen, setSidebarOpen } = useAppStore()

    return (
        <div className="min-h-screen bg-gray-50">
            {/* Mobile sidebar toggle */}
            <div className="lg:hidden fixed top-0 left-0 right-0 z-40 bg-white border-b border-gray-200 px-4 py-3">
                <div className="flex items-center justify-between">
                    <button
                        onClick={() => setSidebarOpen(!sidebarOpen)}
                        className="p-2 rounded-lg hover:bg-gray-100"
                    >
                        {sidebarOpen ? (
                            <XMarkIcon className="w-6 h-6" />
                        ) : (
                            <Bars3Icon className="w-6 h-6" />
                        )}
                    </button>
                    <h1 className="text-lg font-semibold text-primary-600">Medical Quiz Generator</h1>
                    <div className="w-10" />
                </div>
            </div>

            {/* Sidebar */}
            <aside className={clsx(
                'fixed inset-y-0 left-0 z-30 w-64 bg-white border-r border-gray-200 transform transition-transform duration-300 ease-in-out lg:translate-x-0',
                sidebarOpen ? 'translate-x-0' : '-translate-x-full'
            )}>
                <div className="flex flex-col h-full">
                    {/* Logo */}
                    <div className="flex items-center justify-center h-16 px-4 border-b border-gray-200">
                        <Link to="/" className="flex items-center space-x-2">
                            <div className="w-8 h-8 bg-primary-600 rounded-lg flex items-center justify-center">
                                <SparklesIcon className="w-5 h-5 text-white" />
                            </div>
                            <span className="text-lg font-bold text-gray-900">MedQuiz AI</span>
                        </Link>
                    </div>

                    {/* Navigation */}
                    <nav className="flex-1 px-4 py-6 space-y-1 overflow-y-auto">
                        {navigation.map((item) => {
                            const isActive = location.pathname === item.href
                            return (
                                <Link
                                    key={item.name}
                                    to={item.href}
                                    onClick={() => window.innerWidth < 1024 && setSidebarOpen(false)}
                                    className={clsx(
                                        'flex items-center px-4 py-3 text-sm font-medium rounded-lg transition-colors',
                                        isActive
                                            ? 'bg-primary-50 text-primary-700'
                                            : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
                                    )}
                                >
                                    <item.icon className={clsx(
                                        'w-5 h-5 mr-3',
                                        isActive ? 'text-primary-600' : 'text-gray-400'
                                    )} />
                                    {item.name}
                                </Link>
                            )
                        })}
                    </nav>
                </div>
            </aside>

            {/* Overlay for mobile */}
            {sidebarOpen && (
                <div
                    className="fixed inset-0 bg-black bg-opacity-50 z-20 lg:hidden"
                    onClick={() => setSidebarOpen(false)}
                />
            )}

            {/* Main content */}
            <main className={clsx(
                'transition-all duration-300 ease-in-out',
                'lg:ml-64 pt-16 lg:pt-0'
            )}>
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                    {children}
                </div>
            </main>
        </div>
    )
}
