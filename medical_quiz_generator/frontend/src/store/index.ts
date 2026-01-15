import { create } from 'zustand'
import { Document, Question } from '../api'

interface AppState {
    // Documents
    selectedDocuments: string[]
    setSelectedDocuments: (ids: string[]) => void
    toggleDocumentSelection: (id: string) => void
    clearDocumentSelection: () => void

    // Questions
    selectedQuestions: string[]
    setSelectedQuestions: (ids: string[]) => void
    toggleQuestionSelection: (id: string) => void
    clearQuestionSelection: () => void

    // Generation
    currentTaskId: string | null
    setCurrentTaskId: (id: string | null) => void

    // UI State
    sidebarOpen: boolean
    setSidebarOpen: (open: boolean) => void
}

export const useAppStore = create<AppState>((set) => ({
    // Documents
    selectedDocuments: [],
    setSelectedDocuments: (ids) => set({ selectedDocuments: ids }),
    toggleDocumentSelection: (id) => set((state) => ({
        selectedDocuments: state.selectedDocuments.includes(id)
            ? state.selectedDocuments.filter((docId) => docId !== id)
            : [...state.selectedDocuments, id],
    })),
    clearDocumentSelection: () => set({ selectedDocuments: [] }),

    // Questions
    selectedQuestions: [],
    setSelectedQuestions: (ids) => set({ selectedQuestions: ids }),
    toggleQuestionSelection: (id) => set((state) => ({
        selectedQuestions: state.selectedQuestions.includes(id)
            ? state.selectedQuestions.filter((qId) => qId !== id)
            : [...state.selectedQuestions, id],
    })),
    clearQuestionSelection: () => set({ selectedQuestions: [] }),

    // Generation
    currentTaskId: null,
    setCurrentTaskId: (id) => set({ currentTaskId: id }),

    // UI State
    sidebarOpen: true,
    setSidebarOpen: (open) => set({ sidebarOpen: open }),
}))
