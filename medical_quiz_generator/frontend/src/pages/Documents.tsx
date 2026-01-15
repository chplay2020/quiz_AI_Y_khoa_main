import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
    DocumentIcon,
    TrashIcon,
    EyeIcon,
    CheckCircleIcon,
    ClockIcon,
    ExclamationCircleIcon
} from '@heroicons/react/24/outline'
import toast from 'react-hot-toast'
import clsx from 'clsx'
import FileUpload from '../components/FileUpload'
import { documentsApi, Document } from '../api'
import { useAppStore } from '../store'

export default function Documents() {
    const [showUpload, setShowUpload] = useState(false)
    const [selectedDoc, setSelectedDoc] = useState<Document | null>(null)
    const queryClient = useQueryClient()
    const { selectedDocuments, toggleDocumentSelection } = useAppStore()

    const { data, isLoading } = useQuery({
        queryKey: ['documents'],
        queryFn: () => documentsApi.list(),
        refetchInterval: 5000, // Refresh every 5 seconds to check processing status
    })

    const deleteMutation = useMutation({
        mutationFn: documentsApi.delete,
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['documents'] })
            toast.success('Đã xóa tài liệu')
        },
        onError: () => {
            toast.error('Không thể xóa tài liệu')
        },
    })

    const documents = data?.data?.documents || []

    const getStatusIcon = (status: string) => {
        switch (status) {
            case 'completed':
                return <CheckCircleIcon className="w-5 h-5 text-green-500" />
            case 'processing':
                return <ClockIcon className="w-5 h-5 text-yellow-500 animate-spin" />
            case 'failed':
                return <ExclamationCircleIcon className="w-5 h-5 text-red-500" />
            default:
                return <ClockIcon className="w-5 h-5 text-gray-400" />
        }
    }

    const getStatusText = (status: string) => {
        switch (status) {
            case 'completed':
                return 'Hoàn thành'
            case 'processing':
                return 'Đang xử lý'
            case 'failed':
                return 'Thất bại'
            default:
                return 'Chờ xử lý'
        }
    }

    const formatFileSize = (bytes: number) => {
        if (bytes < 1024) return bytes + ' B'
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
        return (bytes / 1024 / 1024).toFixed(2) + ' MB'
    }

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                <div>
                    <h1 className="text-2xl font-bold text-gray-900">Quản lý tài liệu</h1>
                    <p className="text-gray-600 mt-1">Tải lên và quản lý tài liệu y khoa của bạn</p>
                </div>
                <button
                    onClick={() => setShowUpload(!showUpload)}
                    className="btn-primary"
                >
                    {showUpload ? 'Đóng' : 'Tải lên tài liệu mới'}
                </button>
            </div>

            {/* Upload Section */}
            {showUpload && (
                <div className="card">
                    <h2 className="text-lg font-semibold text-gray-900 mb-4">Tải lên tài liệu</h2>
                    <FileUpload onUploadComplete={() => setShowUpload(false)} />
                </div>
            )}

            {/* Documents List */}
            <div className="card">
                <div className="flex items-center justify-between mb-4">
                    <h2 className="text-lg font-semibold text-gray-900">Danh sách tài liệu</h2>
                    {selectedDocuments.length > 0 && (
                        <span className="text-sm text-primary-600 font-medium">
                            Đã chọn {selectedDocuments.length} tài liệu
                        </span>
                    )}
                </div>

                {isLoading ? (
                    <div className="text-center py-12">
                        <div className="animate-spin w-8 h-8 border-4 border-primary-500 border-t-transparent rounded-full mx-auto"></div>
                        <p className="text-gray-500 mt-4">Đang tải...</p>
                    </div>
                ) : documents.length === 0 ? (
                    <div className="text-center py-12">
                        <DocumentIcon className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                        <p className="text-gray-500">Chưa có tài liệu nào</p>
                        <button
                            onClick={() => setShowUpload(true)}
                            className="mt-4 btn-primary"
                        >
                            Tải lên tài liệu đầu tiên
                        </button>
                    </div>
                ) : (
                    <div className="overflow-x-auto">
                        <table className="w-full">
                            <thead>
                                <tr className="border-b border-gray-200">
                                    <th className="px-4 py-3 text-left">
                                        <input
                                            type="checkbox"
                                            className="rounded border-gray-300"
                                            onChange={(e) => {
                                                if (e.target.checked) {
                                                    documents.forEach((doc: Document) => {
                                                        if (!selectedDocuments.includes(doc.id)) {
                                                            toggleDocumentSelection(doc.id)
                                                        }
                                                    })
                                                } else {
                                                    documents.forEach((doc: Document) => {
                                                        if (selectedDocuments.includes(doc.id)) {
                                                            toggleDocumentSelection(doc.id)
                                                        }
                                                    })
                                                }
                                            }}
                                        />
                                    </th>
                                    <th className="px-4 py-3 text-left text-sm font-medium text-gray-500">Tài liệu</th>
                                    <th className="px-4 py-3 text-left text-sm font-medium text-gray-500">Chuyên khoa</th>
                                    <th className="px-4 py-3 text-left text-sm font-medium text-gray-500">Kích thước</th>
                                    <th className="px-4 py-3 text-left text-sm font-medium text-gray-500">Chunks</th>
                                    <th className="px-4 py-3 text-left text-sm font-medium text-gray-500">Trạng thái</th>
                                    <th className="px-4 py-3 text-right text-sm font-medium text-gray-500">Thao tác</th>
                                </tr>
                            </thead>
                            <tbody>
                                {documents.map((doc: Document) => (
                                    <tr
                                        key={doc.id}
                                        className={clsx(
                                            'border-b border-gray-100 hover:bg-gray-50',
                                            selectedDocuments.includes(doc.id) && 'bg-primary-50'
                                        )}
                                    >
                                        <td className="px-4 py-4">
                                            <input
                                                type="checkbox"
                                                className="rounded border-gray-300"
                                                checked={selectedDocuments.includes(doc.id)}
                                                onChange={() => toggleDocumentSelection(doc.id)}
                                            />
                                        </td>
                                        <td className="px-4 py-4">
                                            <div className="flex items-center">
                                                <DocumentIcon className="w-8 h-8 text-gray-400 mr-3" />
                                                <div>
                                                    <p className="font-medium text-gray-900">{doc.title}</p>
                                                    <p className="text-sm text-gray-500">{doc.filename}</p>
                                                </div>
                                            </div>
                                        </td>
                                        <td className="px-4 py-4">
                                            {doc.specialty && (
                                                <span className="px-2 py-1 bg-primary-100 text-primary-700 rounded-full text-xs">
                                                    {doc.specialty}
                                                </span>
                                            )}
                                        </td>
                                        <td className="px-4 py-4 text-sm text-gray-600">
                                            {formatFileSize(doc.file_size)}
                                        </td>
                                        <td className="px-4 py-4 text-sm text-gray-600">
                                            {doc.num_chunks}
                                        </td>
                                        <td className="px-4 py-4">
                                            <div className="flex items-center">
                                                {getStatusIcon(doc.status)}
                                                <span className="ml-2 text-sm">{getStatusText(doc.status)}</span>
                                            </div>
                                        </td>
                                        <td className="px-4 py-4 text-right">
                                            <div className="flex items-center justify-end space-x-2">
                                                <button
                                                    onClick={() => setSelectedDoc(doc)}
                                                    className="p-2 hover:bg-gray-100 rounded-lg"
                                                    title="Xem chi tiết"
                                                >
                                                    <EyeIcon className="w-5 h-5 text-gray-500" />
                                                </button>
                                                <button
                                                    onClick={() => {
                                                        if (confirm('Bạn có chắc muốn xóa tài liệu này?')) {
                                                            deleteMutation.mutate(doc.id)
                                                        }
                                                    }}
                                                    className="p-2 hover:bg-red-100 rounded-lg"
                                                    title="Xóa"
                                                >
                                                    <TrashIcon className="w-5 h-5 text-red-500" />
                                                </button>
                                            </div>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
            </div>

            {/* Document Detail Modal */}
            {selectedDoc && (
                <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
                    <div className="bg-white rounded-xl max-w-2xl w-full max-h-[80vh] overflow-y-auto">
                        <div className="p-6">
                            <div className="flex items-start justify-between mb-4">
                                <h2 className="text-xl font-bold text-gray-900">{selectedDoc.title}</h2>
                                <button
                                    onClick={() => setSelectedDoc(null)}
                                    className="text-gray-500 hover:text-gray-700"
                                >
                                    ✕
                                </button>
                            </div>
                            <div className="space-y-4">
                                <div>
                                    <label className="text-sm font-medium text-gray-500">File</label>
                                    <p className="text-gray-900">{selectedDoc.filename}</p>
                                </div>
                                {selectedDoc.description && (
                                    <div>
                                        <label className="text-sm font-medium text-gray-500">Mô tả</label>
                                        <p className="text-gray-900">{selectedDoc.description}</p>
                                    </div>
                                )}
                                <div className="grid grid-cols-2 gap-4">
                                    <div>
                                        <label className="text-sm font-medium text-gray-500">Loại file</label>
                                        <p className="text-gray-900 uppercase">{selectedDoc.file_type}</p>
                                    </div>
                                    <div>
                                        <label className="text-sm font-medium text-gray-500">Kích thước</label>
                                        <p className="text-gray-900">{formatFileSize(selectedDoc.file_size)}</p>
                                    </div>
                                    <div>
                                        <label className="text-sm font-medium text-gray-500">Số chunks</label>
                                        <p className="text-gray-900">{selectedDoc.num_chunks}</p>
                                    </div>
                                    <div>
                                        <label className="text-sm font-medium text-gray-500">Trạng thái</label>
                                        <p className="text-gray-900">{getStatusText(selectedDoc.status)}</p>
                                    </div>
                                </div>
                                {selectedDoc.tags && selectedDoc.tags.length > 0 && (
                                    <div>
                                        <label className="text-sm font-medium text-gray-500">Tags</label>
                                        <div className="flex flex-wrap gap-2 mt-1">
                                            {selectedDoc.tags.map((tag, i) => (
                                                <span key={i} className="px-2 py-1 bg-gray-100 rounded-full text-sm">
                                                    {tag}
                                                </span>
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}
