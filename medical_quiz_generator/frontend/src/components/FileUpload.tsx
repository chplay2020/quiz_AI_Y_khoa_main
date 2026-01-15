import { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { CloudArrowUpIcon, DocumentIcon, XMarkIcon } from '@heroicons/react/24/outline'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import toast from 'react-hot-toast'
import clsx from 'clsx'
import { documentsApi } from '../api'

interface FileUploadProps {
    onUploadComplete?: () => void
}

interface UploadingFile {
    file: File
    progress: number
    status: 'uploading' | 'success' | 'error'
    error?: string
}

export default function FileUpload({ onUploadComplete }: FileUploadProps) {
    const [uploadingFiles, setUploadingFiles] = useState<UploadingFile[]>([])
    const [metadata, setMetadata] = useState({
        title: '',
        description: '',
        specialty: '',
        tags: '',
    })

    const queryClient = useQueryClient()

    const uploadMutation = useMutation({
        mutationFn: async (file: File) => {
            return documentsApi.upload(file, metadata)
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['documents'] })
            toast.success('Tải lên thành công! Đang xử lý tài liệu...')
            onUploadComplete?.()
        },
        onError: (error: Error) => {
            toast.error(`Lỗi: ${error.message}`)
        },
    })

    const onDrop = useCallback(async (acceptedFiles: File[]) => {
        const newFiles = acceptedFiles.map(file => ({
            file,
            progress: 0,
            status: 'uploading' as const,
        }))

        setUploadingFiles(prev => [...prev, ...newFiles])

        for (const fileInfo of newFiles) {
            try {
                await uploadMutation.mutateAsync(fileInfo.file)
                setUploadingFiles(prev =>
                    prev.map(f =>
                        f.file === fileInfo.file
                            ? { ...f, progress: 100, status: 'success' as const }
                            : f
                    )
                )
            } catch (error) {
                setUploadingFiles(prev =>
                    prev.map(f =>
                        f.file === fileInfo.file
                            ? { ...f, status: 'error' as const, error: (error as Error).message }
                            : f
                    )
                )
            }
        }
    }, [uploadMutation, metadata])

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: {
            'application/pdf': ['.pdf'],
            'application/vnd.openxmlformats-officedocument.presentationml.presentation': ['.pptx'],
            'application/vnd.ms-powerpoint': ['.ppt'],
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
            'application/msword': ['.doc'],
            'text/plain': ['.txt'],
        },
        maxSize: 50 * 1024 * 1024, // 50MB
    })

    const removeFile = (file: File) => {
        setUploadingFiles(prev => prev.filter(f => f.file !== file))
    }

    return (
        <div className="space-y-6">
            {/* Metadata form */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                    <label className="label">Tiêu đề (tùy chọn)</label>
                    <input
                        type="text"
                        className="input"
                        placeholder="Nhập tiêu đề tài liệu"
                        value={metadata.title}
                        onChange={(e) => setMetadata({ ...metadata, title: e.target.value })}
                    />
                </div>
                <div>
                    <label className="label">Chuyên khoa</label>
                    <select
                        className="input"
                        value={metadata.specialty}
                        onChange={(e) => setMetadata({ ...metadata, specialty: e.target.value })}
                    >
                        <option value="">Chọn chuyên khoa</option>
                        <option value="Internal Medicine">Nội khoa</option>
                        <option value="Surgery">Ngoại khoa</option>
                        <option value="Pediatrics">Nhi khoa</option>
                        <option value="Obstetrics & Gynecology">Sản phụ khoa</option>
                        <option value="Cardiology">Tim mạch</option>
                        <option value="Neurology">Thần kinh</option>
                        <option value="Oncology">Ung bướu</option>
                        <option value="Emergency Medicine">Cấp cứu</option>
                        <option value="Pharmacology">Dược lý</option>
                        <option value="Anatomy">Giải phẫu</option>
                        <option value="Physiology">Sinh lý</option>
                        <option value="Biochemistry">Sinh hóa</option>
                        <option value="Microbiology">Vi sinh</option>
                        <option value="Public Health">Y tế công cộng</option>
                    </select>
                </div>
                <div className="md:col-span-2">
                    <label className="label">Mô tả (tùy chọn)</label>
                    <textarea
                        className="input"
                        rows={2}
                        placeholder="Mô tả ngắn về nội dung tài liệu"
                        value={metadata.description}
                        onChange={(e) => setMetadata({ ...metadata, description: e.target.value })}
                    />
                </div>
                <div className="md:col-span-2">
                    <label className="label">Tags (phân cách bằng dấu phẩy)</label>
                    <input
                        type="text"
                        className="input"
                        placeholder="vd: tim mạch, ECG, nhồi máu cơ tim"
                        value={metadata.tags}
                        onChange={(e) => setMetadata({ ...metadata, tags: e.target.value })}
                    />
                </div>
            </div>

            {/* Dropzone */}
            <div
                {...getRootProps()}
                className={clsx(
                    'border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors',
                    isDragActive
                        ? 'border-primary-500 bg-primary-50'
                        : 'border-gray-300 hover:border-primary-400 hover:bg-gray-50'
                )}
            >
                <input {...getInputProps()} />
                <CloudArrowUpIcon className="w-12 h-12 mx-auto text-gray-400" />
                <p className="mt-4 text-lg font-medium text-gray-700">
                    {isDragActive ? 'Thả file vào đây...' : 'Kéo thả file hoặc click để chọn'}
                </p>
                <p className="mt-2 text-sm text-gray-500">
                    Hỗ trợ: PDF, PPTX, DOCX, TXT (Tối đa 50MB)
                </p>
            </div>

            {/* Uploading files list */}
            {uploadingFiles.length > 0 && (
                <div className="space-y-3">
                    <h4 className="font-medium text-gray-700">File đang tải lên</h4>
                    {uploadingFiles.map((fileInfo, index) => (
                        <div
                            key={index}
                            className={clsx(
                                'flex items-center p-4 rounded-lg border',
                                fileInfo.status === 'success' && 'bg-green-50 border-green-200',
                                fileInfo.status === 'error' && 'bg-red-50 border-red-200',
                                fileInfo.status === 'uploading' && 'bg-gray-50 border-gray-200'
                            )}
                        >
                            <DocumentIcon className="w-8 h-8 text-gray-400 mr-3" />
                            <div className="flex-1 min-w-0">
                                <p className="text-sm font-medium text-gray-900 truncate">
                                    {fileInfo.file.name}
                                </p>
                                <p className="text-xs text-gray-500">
                                    {(fileInfo.file.size / 1024 / 1024).toFixed(2)} MB
                                </p>
                                {fileInfo.status === 'uploading' && (
                                    <div className="mt-2 w-full bg-gray-200 rounded-full h-2">
                                        <div
                                            className="bg-primary-600 h-2 rounded-full transition-all"
                                            style={{ width: `${fileInfo.progress}%` }}
                                        />
                                    </div>
                                )}
                                {fileInfo.status === 'error' && (
                                    <p className="text-xs text-red-600 mt-1">{fileInfo.error}</p>
                                )}
                            </div>
                            <button
                                onClick={() => removeFile(fileInfo.file)}
                                className="ml-3 p-1 hover:bg-gray-200 rounded"
                            >
                                <XMarkIcon className="w-5 h-5 text-gray-500" />
                            </button>
                        </div>
                    ))}
                </div>
            )}
        </div>
    )
}
