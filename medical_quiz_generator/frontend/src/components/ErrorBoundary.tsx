import { Component, ReactNode } from 'react'
import { ExclamationTriangleIcon } from '@heroicons/react/24/outline'

interface Props {
    children: ReactNode
}

interface State {
    hasError: boolean
    error: Error | null
}

export default class ErrorBoundary extends Component<Props, State> {
    constructor(props: Props) {
        super(props)
        this.state = { hasError: false, error: null }
    }

    static getDerivedStateFromError(error: Error): State {
        return { hasError: true, error }
    }

    componentDidCatch(error: Error, errorInfo: any) {
        console.error('Error caught by boundary:', error, errorInfo)
    }

    render() {
        if (this.state.hasError) {
            return (
                <div className="min-h-screen flex items-center justify-center bg-gray-50 px-4">
                    <div className="max-w-md w-full bg-white rounded-lg shadow-lg p-8 text-center">
                        <ExclamationTriangleIcon className="w-16 h-16 text-red-500 mx-auto mb-4" />
                        <h1 className="text-2xl font-bold text-gray-900 mb-2">
                            Oops! Có lỗi xảy ra
                        </h1>
                        <p className="text-gray-600 mb-4">
                            {this.state.error?.message || 'Đã xảy ra lỗi không mong muốn'}
                        </p>
                        <div className="space-y-3">
                            <button
                                onClick={() => window.location.reload()}
                                className="btn-primary w-full"
                            >
                                Tải lại trang
                            </button>
                            <button
                                onClick={() => window.location.href = '/'}
                                className="btn-secondary w-full"
                            >
                                Về trang chủ
                            </button>
                        </div>
                        {import.meta.env.DEV && (
                            <details className="mt-6 text-left">
                                <summary className="text-sm text-gray-500 cursor-pointer">
                                    Chi tiết lỗi (Dev only)
                                </summary>
                                <pre className="mt-2 text-xs text-red-600 bg-red-50 p-3 rounded overflow-auto">
                                    {this.state.error?.stack}
                                </pre>
                            </details>
                        )}
                    </div>
                </div>
            )
        }

        return this.props.children
    }
}
