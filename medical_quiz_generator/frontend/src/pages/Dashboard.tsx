import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import {
    DocumentTextIcon,
    QuestionMarkCircleIcon,
    SparklesIcon,
    ArrowTrendingUpIcon
} from '@heroicons/react/24/outline'
import { documentsApi, questionsApi } from '../api'

export default function Dashboard() {
    const { data: docStats } = useQuery({
        queryKey: ['document-stats'],
        queryFn: documentsApi.getStats,
    })

    const { data: questionStats } = useQuery({
        queryKey: ['question-stats'],
        queryFn: questionsApi.getStats,
    })

    const stats = [
        {
            name: 'Tổng tài liệu',
            value: docStats?.data?.total_documents || 0,
            icon: DocumentTextIcon,
            color: 'bg-blue-500',
            link: '/documents',
        },
        {
            name: 'Chunks đã xử lý',
            value: docStats?.data?.total_chunks || 0,
            icon: ArrowTrendingUpIcon,
            color: 'bg-green-500',
            link: '/documents',
        },
        {
            name: 'Câu hỏi đã tạo',
            value: questionStats?.data?.total_questions || 0,
            icon: QuestionMarkCircleIcon,
            color: 'bg-purple-500',
            link: '/questions',
        },
    ]

    return (
        <div className="space-y-8">
            {/* Welcome Section */}
            <div className="bg-gradient-to-r from-primary-600 to-primary-800 rounded-2xl p-8 text-white">
                <h1 className="text-3xl font-bold mb-2">Medical Quiz Generator</h1>
                <p className="text-primary-100 text-lg mb-6">
                    Tạo câu hỏi trắc nghiệm y khoa tự động từ tài liệu với AI
                </p>
                <div className="flex flex-wrap gap-4">
                    <Link to="/documents" className="bg-white text-primary-700 hover:bg-primary-50 font-medium py-2 px-6 rounded-lg transition-colors">
                        Tải lên tài liệu
                    </Link>
                    <Link to="/generate" className="bg-primary-500 hover:bg-primary-400 text-white font-medium py-2 px-6 rounded-lg transition-colors border border-primary-400">
                        Tạo câu hỏi mới
                    </Link>
                </div>
            </div>

            {/* Stats Grid */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {stats.map((stat) => (
                    <Link
                        key={stat.name}
                        to={stat.link}
                        className="card hover:shadow-md transition-shadow"
                    >
                        <div className="flex items-center">
                            <div className={`${stat.color} p-3 rounded-lg`}>
                                <stat.icon className="w-6 h-6 text-white" />
                            </div>
                            <div className="ml-4">
                                <p className="text-sm font-medium text-gray-500">{stat.name}</p>
                                <p className="text-2xl font-bold text-gray-900">{stat.value}</p>
                            </div>
                        </div>
                    </Link>
                ))}
            </div>

            {/* How it works */}
            <div className="card">
                <h2 className="text-xl font-bold text-gray-900 mb-6">Cách sử dụng</h2>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                    <div className="text-center">
                        <div className="w-12 h-12 bg-primary-100 rounded-full flex items-center justify-center mx-auto mb-4">
                            <span className="text-xl font-bold text-primary-600">1</span>
                        </div>
                        <h3 className="font-semibold text-gray-900 mb-2">Tải lên tài liệu</h3>
                        <p className="text-sm text-gray-600">
                            Upload slide, PDF, hoặc tài liệu guideline y khoa của bạn
                        </p>
                    </div>
                    <div className="text-center">
                        <div className="w-12 h-12 bg-primary-100 rounded-full flex items-center justify-center mx-auto mb-4">
                            <span className="text-xl font-bold text-primary-600">2</span>
                        </div>
                        <h3 className="font-semibold text-gray-900 mb-2">AI xử lý & tạo câu hỏi</h3>
                        <p className="text-sm text-gray-600">
                            Hệ thống RAG phân tích nội dung và AI tạo câu hỏi chất lượng cao
                        </p>
                    </div>
                    <div className="text-center">
                        <div className="w-12 h-12 bg-primary-100 rounded-full flex items-center justify-center mx-auto mb-4">
                            <span className="text-xl font-bold text-primary-600">3</span>
                        </div>
                        <h3 className="font-semibold text-gray-900 mb-2">Xuất & sử dụng</h3>
                        <p className="text-sm text-gray-600">
                            Xem, chỉnh sửa và xuất bộ câu hỏi theo định dạng mong muốn
                        </p>
                    </div>
                </div>
            </div>

            {/* Features */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="card">
                    <div className="flex items-center mb-4">
                        <SparklesIcon className="w-6 h-6 text-primary-600 mr-2" />
                        <h3 className="text-lg font-semibold text-gray-900">Tính năng nổi bật</h3>
                    </div>
                    <ul className="space-y-3 text-sm text-gray-600">
                        <li className="flex items-center">
                            <span className="w-2 h-2 bg-primary-500 rounded-full mr-2"></span>
                            Hỗ trợ nhiều định dạng: PDF, PPTX, DOCX
                        </li>
                        <li className="flex items-center">
                            <span className="w-2 h-2 bg-primary-500 rounded-full mr-2"></span>
                            RAG-based cho độ chính xác cao
                        </li>
                        <li className="flex items-center">
                            <span className="w-2 h-2 bg-primary-500 rounded-full mr-2"></span>
                            Tạo câu hỏi theo độ khó & chủ đề
                        </li>
                        <li className="flex items-center">
                            <span className="w-2 h-2 bg-primary-500 rounded-full mr-2"></span>
                            Câu hỏi tình huống lâm sàng (Case-based)
                        </li>
                        <li className="flex items-center">
                            <span className="w-2 h-2 bg-primary-500 rounded-full mr-2"></span>
                            Xuất nhiều định dạng: JSON, PDF, Excel
                        </li>
                    </ul>
                </div>

                <div className="card">
                    <div className="flex items-center mb-4">
                        <DocumentTextIcon className="w-6 h-6 text-primary-600 mr-2" />
                        <h3 className="text-lg font-semibold text-gray-900">Chuyên khoa hỗ trợ</h3>
                    </div>
                    <div className="flex flex-wrap gap-2">
                        {['Nội khoa', 'Ngoại khoa', 'Nhi khoa', 'Sản phụ khoa', 'Tim mạch',
                            'Thần kinh', 'Giải phẫu', 'Sinh lý', 'Dược lý', 'Vi sinh'].map((specialty) => (
                                <span
                                    key={specialty}
                                    className="px-3 py-1 bg-gray-100 text-gray-700 rounded-full text-sm"
                                >
                                    {specialty}
                                </span>
                            ))}
                        <span className="px-3 py-1 bg-primary-100 text-primary-700 rounded-full text-sm">
                            + nhiều hơn...
                        </span>
                    </div>
                </div>
            </div>
        </div>
    )
}
